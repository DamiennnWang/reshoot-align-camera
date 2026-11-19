package com.fieldalign.camera

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.fieldalign.camera.ui.AppTheme
import com.fieldalign.camera.ui.CameraScreen
import com.fieldalign.camera.ui.CompareScreen
import com.fieldalign.camera.ui.HomeScreen
import com.fieldalign.camera.ui.RegisterScreen
import com.fieldalign.camera.ui.UnclassifiedScreen
import com.fieldalign.camera.ui.ZoneScreen
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    private val vm: AppViewModel by viewModels()
    private var hasCamera by mutableStateOf(false)

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { hasCamera = it }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        handleShare(intent)
    }

    /** 其他应用“分享”过来的照片：直接走导入流程。 */
    private fun handleShare(i: android.content.Intent?) {
        i ?: return
        val uris: List<android.net.Uri> = when (i.action) {
            android.content.Intent.ACTION_SEND -> listOfNotNull(
                androidx.core.content.IntentCompat.getParcelableExtra(i, android.content.Intent.EXTRA_STREAM, android.net.Uri::class.java))
            android.content.Intent.ACTION_SEND_MULTIPLE ->
                androidx.core.content.IntentCompat.getParcelableArrayListExtra(i, android.content.Intent.EXTRA_STREAM, android.net.Uri::class.java) ?: emptyList()
            else -> emptyList()
        }
        if (uris.isNotEmpty()) { vm.importUris(uris); i.action = null }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val cvOk = OpenCVLoader.initLocal()
        hasCamera = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        handleShare(intent)
        setContent {
            AppTheme {
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    if (!cvOk) Text("OpenCV 加载失败，请确认安装的是 arm64 手机版本", Modifier.padding(24.dp))
                    else AppRoot(vm, hasCamera) { askCamera.launch(Manifest.permission.CAMERA) }
                }
            }
        }
    }
}

@Composable
fun AppRoot(vm: AppViewModel, hasCamera: Boolean, requestCamera: () -> Unit) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    BackHandler(enabled = vm.stack.size > 1) { vm.back() }
    LaunchedEffect(vm.toast) {
        vm.toast?.let { Toast.makeText(ctx, it, Toast.LENGTH_LONG).show(); vm.toast = null }
    }
    Box(Modifier.fillMaxSize()) {
        when (val s = vm.current) {
            is Screen.Home -> HomeScreen(vm)
            is Screen.Camera ->
                if (hasCamera) CameraScreen(vm, s.zoneId)
                else Column(Modifier.fillMaxSize().padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    LaunchedEffect(Unit) { requestCamera() }   // 进入相机页时直接弹出系统授权
                    Text("需要相机权限才能拍照")
                    Button(onClick = requestCamera, modifier = Modifier.padding(top = 16.dp)) { Text("授权相机") }
                }
            is Screen.ZoneDetail -> ZoneScreen(vm, s.zoneId)
            is Screen.Compare -> CompareScreen(vm, s.photoId)
            is Screen.Register -> RegisterScreen(vm, s)
            is Screen.Unclassified -> UnclassifiedScreen(vm, s.outcome)
        }
        if (vm.processing) {
            Box(Modifier.fillMaxSize().padding(0.dp), contentAlignment = Alignment.Center) {
                Surface(color = Color(0xCC000000), shape = MaterialTheme.shapes.medium) {
                    Column(Modifier.padding(28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color.White)
                        Text("识别、对齐、对比中…", color = Color.White, modifier = Modifier.padding(top = 12.dp))
                        vm.importProgress?.let { Text(it, color = Color.LightGray, modifier = Modifier.padding(top = 4.dp)) }
                    }
                }
            }
        }
    }
}
