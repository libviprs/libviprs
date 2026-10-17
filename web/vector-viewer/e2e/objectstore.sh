#!/usr/bin/env bash
# Runs e2e/objectstore.spec.ts against a real S3-compatible object store.
#
# Runs on a docker host and never on a laptop. It starts RustFS in a throw-away
# container on a throw-away network, makes a bucket, sets the CORS rules a
# browser deployment needs, makes the object readable without credentials,
# uploads the fixture archive, runs the spec in the Playwright image on the same
# network, and removes everything it made. The credentials below are made up
# for this one container and protect nothing.
#
# usage: objectstore.sh <path to this viewer directory on the docker host>
set -euo pipefail

TREE=${1:?viewer directory}
ID=$$
NET=viprs-os-$ID
STORE=viprs-rustfs-$ID
KEY=viprs
SECRET=viprs-throwaway-secret
BUCKET=drawings
PW=mcr.microsoft.com/playwright:v1.64.0-noble

cleanup() {
  docker rm -f "$STORE" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create "$NET" >/dev/null
docker run -d --name "$STORE" --network "$NET" --platform linux/amd64 \
  -e RUSTFS_ACCESS_KEY=$KEY -e RUSTFS_SECRET_KEY=$SECRET \
  rustfs/rustfs:latest >/dev/null

aws() {
  docker run --rm --platform linux/amd64 --network "$NET" \
    -e AWS_ACCESS_KEY_ID=$KEY -e AWS_SECRET_ACCESS_KEY=$SECRET -e AWS_DEFAULT_REGION=us-east-1 \
    -v "$TREE/e2e:/e2e:ro" amazon/aws-cli "$@" --endpoint-url "http://$STORE:9000"
}

for i in $(seq 1 60); do
  if aws s3 ls >/dev/null 2>&1; then break; fi
  sleep 1
done
aws s3 ls >/dev/null

aws s3 mb "s3://$BUCKET"
cat > /tmp/cors-$ID.json <<'JSON'
{"CORSRules":[{"AllowedOrigins":["*"],"AllowedMethods":["GET","HEAD"],"AllowedHeaders":["*"],"ExposeHeaders":["Content-Range","Content-Length","Accept-Ranges","ETag"],"MaxAgeSeconds":600}]}
JSON
cat > /tmp/policy-$ID.json <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":["s3:GetObject"],"Resource":["arn:aws:s3:::$BUCKET/*"]}]}
JSON
docker run --rm --platform linux/amd64 --network "$NET" \
  -e AWS_ACCESS_KEY_ID=$KEY -e AWS_SECRET_ACCESS_KEY=$SECRET -e AWS_DEFAULT_REGION=us-east-1 \
  -v /tmp/cors-$ID.json:/cors.json:ro -v /tmp/policy-$ID.json:/policy.json:ro \
  -v "$TREE/e2e:/e2e:ro" --entrypoint sh amazon/aws-cli -ec "
    aws --endpoint-url http://$STORE:9000 s3api put-bucket-cors --bucket $BUCKET --cors-configuration file:///cors.json
    aws --endpoint-url http://$STORE:9000 s3api put-bucket-policy --bucket $BUCKET --policy file:///policy.json
    aws --endpoint-url http://$STORE:9000 s3 cp /e2e/fixtures/golden.pmtiles s3://$BUCKET/golden.pmtiles
    aws --endpoint-url http://$STORE:9000 s3api get-bucket-cors --bucket $BUCKET
  "
rm -f /tmp/cors-$ID.json /tmp/policy-$ID.json

docker run --rm --platform linux/amd64 --ipc=host --network "$NET" \
  -e OBJECT_STORE_URL="http://$STORE:9000/$BUCKET/golden.pmtiles" -e LIBVIPRS_REQUIRE_OBJECT_STORE=1 \
  -v "$TREE:/work" -w /work "$PW" npx playwright test e2e/objectstore.spec.ts
