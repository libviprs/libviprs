# The first DWG to PDF exporter for the N5.7 differential (issue #1405): GNU LibreDWG's dwg2SVG,
# then rsvg-convert for the SVG to PDF step, then qpdf for the optional page rotation.
#
# LibreDWG is GPL-3.0-or-later. It lives only in this image and meets libviprs through files, the
# same process boundary pvec-oracle-libredwg uses. Inputs are pinned: the base image by digest and
# the release tarball by sha256 (the build stops if it does not match), the same pins as that image.
# The apt packages of the runtime stage are not version-pinned; export.sh records what they were.
ARG BASE=debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

FROM ${BASE} AS build
ARG LIBREDWG_VERSION=0.14
ARG LIBREDWG_SHA256=62ebb73b984f865960f20ed26619ea5f8789d5e3fd088fa40a2598384da81275
ARG LIBREDWG_URL=https://ftp.gnu.org/gnu/libredwg/libredwg-${LIBREDWG_VERSION}.tar.xz
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl pkg-config xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o libredwg.tar.xz "${LIBREDWG_URL}" \
    && echo "${LIBREDWG_SHA256}  libredwg.tar.xz" | sha256sum -c - \
    && mkdir libredwg && tar -xJf libredwg.tar.xz -C libredwg --strip-components=1
RUN cd /src/libredwg && CFLAGS="-O2 -ffp-contract=off" ./configure --prefix=/opt/stock \
        --disable-shared --enable-static --disable-bindings --disable-python \
        --disable-docs --disable-werror --disable-trace \
    && make -j"$(nproc)" && make install && /opt/stock/bin/dwg2SVG --version

FROM ${BASE} AS runtime
ARG LIBREDWG_VERSION=0.14
RUN apt-get update && apt-get install -y --no-install-recommends librsvg2-bin qpdf \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/stock/bin/dwg2SVG /usr/local/bin/dwg2SVG
COPY --from=build /src/libredwg.tar.xz /usr/share/doc/viprs-dwg-pdf-export/source/libredwg-${LIBREDWG_VERSION}.tar.xz
COPY --from=build /src/libredwg/COPYING /usr/share/doc/viprs-dwg-pdf-export/COPYING.libredwg
ENV LIBREDWG_VERSION=${LIBREDWG_VERSION}
