package com.fieldalign.camera.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/** 一个“系列”（测区）：有二维码时用码内容识别，无码时靠参考照片特征识别。 */
@Entity(tableName = "zone", indices = [Index(value = ["qrText"], unique = true)])
data class Zone(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val qrText: String?,
    val name: String,
    val createdAt: Long,
    val refPhotoId: Long? = null,
    val lat: Double? = null,            // 系列位置（参考照片拍摄时的 GPS），无码时辅助识别
    val lon: Double? = null,
    val azimuth: Double? = null,        // 参考照片拍摄时镜头朝向（度，北=0 顺时针），同一点位不同角度的系列靠它区分
    val pitch: Double? = null,          // 镜头俯仰角（度，水平=0，向上为正）
    val qrMm: Double? = null,           // 二维码打印边长（毫米），用于像素 → 毫米换算
    val kinds: String? = null,          // 测点病害类型（Defects.Kind 代码，逗号分隔）；空 = 通用
)

@Entity(tableName = "photo", indices = [Index("zoneId")])
data class Photo(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val zoneId: Long,
    val path: String,
    val takenAt: Long,
    val status: String,                 // reference / confirmed / markers_only / conflict / dot_moved / features_only / failed
    val message: String,
    val markersJson: String,            // 二维码、圆点（参考照片含选定圆点）
    val alignJson: String? = null,      // 对齐矩阵与指标
    val refPhotoId: Long? = null,       // 对比时用的参考照片
    val changedRatio: Double? = null,
    val changeRegions: Int? = null,
    val similarity: Double? = null,     // 对比区整体相似度（相关系数）
    val lightRatio: Double? = null,     // 明暗变化（阴影/水渍等）占比
    val lightRegions: Int? = null,
    val glareRatio: Double? = null,     // 反光/过曝未参与比较的占比
    val quality: String? = null,        // 拍摄质量提示（多条以换行分隔）
    val alignedPath: String? = null,
    val changesPath: String? = null,
    val how: String? = null,            // 归类方式：二维码 / 图像特征 / 手动
    val lat: Double? = null,            // 拍摄位置（手机 GPS 或照片 EXIF）
    val lon: Double? = null,
    val locAcc: Double? = null,         // 定位精度（米）
    val azimuth: Double? = null,        // 镜头朝向（度）；指南针未校准时为空
    val pitch: Double? = null,          // 镜头俯仰角（度）
    val mmPerPx: Double? = null,        // 参考坐标下每像素对应的毫米数（由参考照片中的二维码尺寸算出）
    val crackCount: Int? = null,        // 裂缝条数
    val crackLenPx: Double? = null,     // 裂缝总长（像素，参考坐标）
    val crackMaxWPx: Double? = null,    // 最大宽度（像素）
    val crackGrowPx: Double? = null,    // 比参考新增的裂缝长度（像素）
    val crackJson: String? = null,      // 每条裂缝的明细 + 参考照片的裂缝统计
    val conclusion: String? = null,     // 文字结论
    val findingsJson: String? = null,   // 病害分析明细（面状/位移/标记线/部件 + 标注图路径），见 Analysis.toJson
)

data class ZoneSummary(
    val id: Long, val qrText: String?, val name: String, val refPhotoId: Long?,
    val photoCount: Int, val lastTaken: Long?, val refPath: String?,
    val lat: Double? = null, val lon: Double? = null,
    val lastStatus: String? = null, val lastChanges: Int? = null,
    val azimuth: Double? = null, val pitch: Double? = null,
    val kinds: String? = null, val lastConclusion: String? = null,
)

@Dao
interface ZoneDao {
    @Query("""SELECT z.id, z.qrText, z.name, z.refPhotoId, z.lat, z.lon, z.azimuth, z.pitch, z.kinds,
              (SELECT conclusion FROM photo p WHERE p.zoneId = z.id AND p.id != z.refPhotoId ORDER BY takenAt DESC LIMIT 1) AS lastConclusion,
              (SELECT status FROM photo p WHERE p.zoneId = z.id AND p.id != z.refPhotoId ORDER BY takenAt DESC LIMIT 1) AS lastStatus,
              (SELECT changeRegions FROM photo p WHERE p.zoneId = z.id AND p.id != z.refPhotoId ORDER BY takenAt DESC LIMIT 1) AS lastChanges,
              (SELECT COUNT(*) FROM photo p WHERE p.zoneId = z.id) AS photoCount,
              (SELECT MAX(takenAt) FROM photo p WHERE p.zoneId = z.id) AS lastTaken,
              (SELECT path FROM photo p WHERE p.id = z.refPhotoId) AS refPath
              FROM zone z ORDER BY lastTaken DESC""")
    fun summaries(): Flow<List<ZoneSummary>>

    @Query("SELECT * FROM zone ORDER BY id") suspend fun all(): List<Zone>
    @Query("SELECT * FROM zone WHERE id = :id") suspend fun get(id: Long): Zone?
    @Query("SELECT * FROM zone WHERE id = :id") fun observe(id: Long): Flow<Zone?>
    @Query("SELECT * FROM zone WHERE qrText = :qr") suspend fun byQr(qr: String): Zone?
    @Insert suspend fun insert(z: Zone): Long
    @Update suspend fun update(z: Zone)
    @Query("DELETE FROM zone WHERE id = :id") suspend fun delete(id: Long)
}

@Dao
interface PhotoDao {
    @Query("SELECT * FROM photo WHERE zoneId = :zoneId ORDER BY takenAt DESC")
    fun forZone(zoneId: Long): Flow<List<Photo>>
    @Query("SELECT * FROM photo WHERE zoneId = :zoneId") suspend fun listForZone(zoneId: Long): List<Photo>
    @Query("SELECT * FROM photo WHERE id = :id") suspend fun get(id: Long): Photo?
    @Insert suspend fun insert(p: Photo): Long
    @Update suspend fun update(p: Photo)
    @Query("DELETE FROM photo WHERE id = :id") suspend fun delete(id: Long)
}

@Database(entities = [Zone::class, Photo::class], version = 7, exportSchema = false)
abstract class AppDb : RoomDatabase() {
    abstract fun zones(): ZoneDao
    abstract fun photos(): PhotoDao

    companion object {
        @Volatile private var inst: AppDb? = null
        /** 3 → 4：增加反光占比和拍摄质量提示两列，保留已有数据。 */
        val MIGRATION_3_4 = object : androidx.room.migration.Migration(3, 4) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE photo ADD COLUMN glareRatio REAL")
                db.execSQL("ALTER TABLE photo ADD COLUMN quality TEXT")
            }
        }
        /** 4 → 5：增加拍摄位置（GPS），用于无码时按位置 + 特征识别系列。 */
        val MIGRATION_4_5 = object : androidx.room.migration.Migration(4, 5) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE zone ADD COLUMN lat REAL")
                db.execSQL("ALTER TABLE zone ADD COLUMN lon REAL")
                db.execSQL("ALTER TABLE photo ADD COLUMN lat REAL")
                db.execSQL("ALTER TABLE photo ADD COLUMN lon REAL")
                db.execSQL("ALTER TABLE photo ADD COLUMN locAcc REAL")
            }
        }
        /** 5 → 6：镜头朝向、毫米比例、裂缝参数、文字结论。 */
        val MIGRATION_5_6 = object : androidx.room.migration.Migration(5, 6) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                for (c in listOf("azimuth REAL", "pitch REAL", "qrMm REAL")) db.execSQL("ALTER TABLE zone ADD COLUMN $c")
                for (c in listOf("azimuth REAL", "pitch REAL", "mmPerPx REAL", "crackCount INTEGER", "crackLenPx REAL",
                    "crackMaxWPx REAL", "crackGrowPx REAL", "crackJson TEXT", "conclusion TEXT"))
                    db.execSQL("ALTER TABLE photo ADD COLUMN $c")
            }
        }
        /** 6 → 7：测点病害类型、病害分析明细。 */
        val MIGRATION_6_7 = object : androidx.room.migration.Migration(6, 7) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE zone ADD COLUMN kinds TEXT")
                db.execSQL("ALTER TABLE photo ADD COLUMN findingsJson TEXT")
            }
        }
        fun get(ctx: Context): AppDb = inst ?: synchronized(this) {
            inst ?: Room.databaseBuilder(ctx.applicationContext, AppDb::class.java, "fieldalign.db")
                .addMigrations(MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7)
                .fallbackToDestructiveMigration()   // 更早的开发版本：重建
                .build().also { inst = it }
        }
    }
}
