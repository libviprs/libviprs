# Android emulator plus the SDK, for the MapLibre Native conformance run (issue #1037).
# Build: docker build -t g6-android:api34 conformance/android   (about 5 GB, mostly the system image)
FROM node:22-bookworm-slim
ENV ANDROID_HOME=/opt/android ANDROID_SDK_ROOT=/opt/android DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/android/cmdline-tools/latest/bin:/opt/android/platform-tools:/opt/android/emulator:/opt/gradle/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless curl unzip ca-certificates procps rsync \
      libpulse0 libnss3 libxcursor1 libxdamage1 libxi6 libxtst6 libasound2 libgl1 libxkbfile1 \
      libx11-xcb1 libdbus-1-3 libxcomposite1 libxrandr2 libxfixes3 libxrender1 libxext6 libbz2-1.0 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/android/cmdline-tools /tmp/dl \
 && curl -fsSL -o /tmp/dl/cmdline.zip https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip \
 && unzip -q /tmp/dl/cmdline.zip -d /tmp/dl && mv /tmp/dl/cmdline-tools /opt/android/cmdline-tools/latest \
 && curl -fsSL -o /tmp/dl/gradle.zip https://services.gradle.org/distributions/gradle-8.7-bin.zip \
 && unzip -q /tmp/dl/gradle.zip -d /opt && mv /opt/gradle-8.7 /opt/gradle && rm -rf /tmp/dl
RUN yes | sdkmanager --licenses >/dev/null \
 && sdkmanager "platform-tools" "emulator" "platforms;android-34" "build-tools;34.0.0" \
      "system-images;android-34;default;x86_64" \
 && echo no | avdmanager create avd -n g6 -k "system-images;android-34;default;x86_64" -d pixel_5
