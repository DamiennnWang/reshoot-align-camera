plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.fieldalign.camera"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.fieldalign.camera"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a") }   // 只打 64 位真机，安装包更小
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    // 设备端算法测试使用 python/synth_test.py 生成的合成照片和真值
    sourceSets.getByName("androidTest").assets.srcDir(layout.buildDirectory.dir("generated/testSamples"))
}

val copySamples by tasks.registering(Copy::class) {
    from("../../samples") { include("*.jpg", "truth.json", "series/*.jpg", "distortion/*.jpg", "distortion/truth.json", "lighting/*.jpg", "lighting/truth.json", "robustness/*.jpg", "robustness/truth.json", "measure/*_ref.jpg", "measure/*_new.jpg", "measure/truth.json")
        exclude("robustness/*_changes.jpg") }
    into(layout.buildDirectory.dir("generated/testSamples/samples"))
}
tasks.configureEach {
    if (name.contains("AndroidTest") && name.contains("Assets")) dependsOn(copySamples)
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.3")
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // 二维码：模型随应用打包，离线可用
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    // 圆点检测、单应矩阵、ORB 特征、差分
    implementation("org.opencv:opencv:4.10.0")

    val room = "2.6.1"
    implementation("androidx.room:room-runtime:$room")
    implementation("androidx.room:room-ktx:$room")
    ksp("androidx.room:room-compiler:$room")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
