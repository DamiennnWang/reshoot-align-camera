"""宣传片三维镜头。用法：
  Blender -b -P scene.py -- <shot> <h|v> [--still F] [--frames A-B]
shot: tunnel / poses / scan / phones / end
输出：render/<shot>_<h|v>/####.png（--still 时输出 render/still_<shot>_<ori>_F.png）

衬砌上的样板墙面贴花就是 samples/promo 两张照片所拍的那块墙（tex/wall_*.png，400×275 mm），
所以片中隧道里的裂缝、二维码、标记点，和 App 实际处理的照片是同一处。
"""
import bpy, bmesh, math, os, sys
from mathutils import Vector, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "tex")
SHOTS_DIR = os.path.join(HERE, "..", "素材")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["tunnel", "h"]
SHOT, ORI = argv[0], argv[1]
STILL = int(argv[argv.index("--still") + 1]) if "--still" in argv else None
FRAMES = argv[argv.index("--frames") + 1] if "--frames" in argv else None
FPS = 30
DUR = dict(tunnel=180, poses=105, scan=180, phones=180, end=180)[SHOT]

# 墙面样板贴花的位置：右侧边墙（x=+4.5，朝 -X），中心 y=60, z=1.4
PATCH_W, PATCH_H = 0.400, 0.275
PATCH_C = Vector((4.497, 60.0, 1.40))
TUN_HALF, WALL_H, Y0, Y1 = 4.5, 2.0, -10.0, 140.0


def tex_uv(u, v):
    """贴图坐标（0..1，左上原点）→ 世界坐标。图像向右 = 世界 -Y，向下 = 世界 -Z。"""
    return Vector((PATCH_C.x, PATCH_C.y + (0.5 - u) * PATCH_W, PATCH_C.z + (0.5 - v) * PATCH_H))


CRACK_TOP = tex_uv(800 / 3200, 300 / 2200)
CRACK_C = tex_uv(870 / 3200, 1090 / 2200)
QR_C = tex_uv(0.52, 0.51)

# ---------------------------------------------------------------- 场景基础
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, DUR
W, H = (1920, 1080) if ORI == "h" else (1080, 1920)
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
ee = sc.eevee
ee.taa_render_samples = 48
for k, v in (("use_volumetric_shadows", True), ("volumetric_tile_size", "4"), ("volumetric_end", 90.0),
             ("volumetric_samples", 64), ("use_raytracing", True), ("use_shadows", True)):
    try:
        setattr(ee, k, v)
    except Exception as e:
        print("eevee", k, e)
sc.view_settings.view_transform = "AgX"
for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
    try:
        sc.view_settings.look = look
        break
    except Exception:
        pass
sc.render.use_motion_blur = SHOT in ("tunnel", "end")


def mat_new(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links


def node(nodes, kind, loc=(0, 0), **kw):
    n = nodes.new(kind)
    n.location = loc
    for k, v in kw.items():
        if k in n.inputs:
            n.inputs[k].default_value = v
        else:
            setattr(n, k, v)
    return n


def bsdf_of(nodes):
    return nodes["Principled BSDF"]


def img(name, noncolor=False):
    im = bpy.data.images.load(os.path.join(TEX, name) if not os.path.isabs(name) else name, check_existing=True)
    if noncolor:
        im.colorspace_settings.name = "Non-Color"
    return im


def link_obj(ob):
    sc.collection.objects.link(ob)
    return ob


def mesh_obj(name, verts, faces, mat=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = link_obj(bpy.data.objects.new(name, me))
    if mat:
        ob.data.materials.append(mat)
    return ob


def box(name, center, size, mat=None, bevel=0.0):
    sx, sy, sz = (s / 2 for s in size)
    v = [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ob = mesh_obj(name, v, f, mat)
    ob.location = center
    if bevel:
        m = ob.modifiers.new("bev", "BEVEL")
        m.width, m.segments = bevel, 3
    return ob


def key(obj, path, frame, value, interp="BEZIER"):
    setattr(obj, path, value) if not isinstance(path, tuple) else None
    obj.keyframe_insert(data_path=path, frame=frame)
    fc = obj.animation_data.action.fcurves if hasattr(obj.animation_data.action, "fcurves") else None
    return fc


def set_key(obj, path, frame, value):
    if "." in path or "[" in path:
        exec(f"obj.{path} = value")
    else:
        setattr(obj, path, value)
    obj.keyframe_insert(data_path=path.split("[")[0] if path.endswith("]") is False else path, frame=frame)


def kf(owner, attr, frame, value):
    """给属性打关键帧（owner 可以是对象、灯光数据、节点输入）。"""
    setattr(owner, attr, value)
    owner.keyframe_insert(data_path=attr, frame=frame)


# ---------------------------------------------------------------- 材质
def concrete_mat():
    m, n, l = mat_new("lining")
    b = bsdf_of(n)
    b.location = (600, 0)
    tc = node(n, "ShaderNodeTexCoord", (-1400, 0))
    sep = node(n, "ShaderNodeSeparateXYZ", (-1200, -300))
    l.new(tc.outputs["Object"], sep.inputs[0])
    big = node(n, "ShaderNodeTexNoise", (-1000, 200), Scale=0.6, Detail=6.0, Roughness=0.55)
    fine = node(n, "ShaderNodeTexNoise", (-1000, -50), Scale=9.0, Detail=12.0, Roughness=0.62)
    l.new(tc.outputs["Object"], big.inputs["Vector"])
    l.new(tc.outputs["Object"], fine.inputs["Vector"])
    # 渗水痕：竖向拉长的噪声
    mp = node(n, "ShaderNodeMapping", (-1200, 400))
    mp.inputs["Scale"].default_value = (1.0, 1.2, 0.08)
    l.new(tc.outputs["Object"], mp.inputs["Vector"])
    streak = node(n, "ShaderNodeTexNoise", (-1000, 450), Scale=3.0, Detail=4.0)
    l.new(mp.outputs["Vector"], streak.inputs["Vector"])
    sramp = node(n, "ShaderNodeValToRGB", (-800, 450))
    sramp.color_ramp.elements[0].position, sramp.color_ramp.elements[1].position = 0.55, 0.75
    sramp.color_ramp.elements[0].color, sramp.color_ramp.elements[1].color = (1, 1, 1, 1), (0.55, 0.52, 0.48, 1)
    l.new(streak.outputs["Fac"], sramp.inputs["Fac"])
    grain = node(n, "ShaderNodeTexNoise", (-1000, -600), Scale=320.0, Detail=2.0)
    l.new(tc.outputs["Object"], grain.inputs["Vector"])
    mix0 = node(n, "ShaderNodeMix", (-800, -100), data_type="FLOAT")
    mix0.inputs["Factor"].default_value = 0.45
    l.new(fine.outputs["Fac"], mix0.inputs["A"])
    l.new(grain.outputs["Fac"], mix0.inputs["B"])
    fine_out = mix0.outputs["Result"]
    mix1 = node(n, "ShaderNodeMix", (-600, 150), data_type="FLOAT")
    mix1.inputs["Factor"].default_value = 0.35
    l.new(big.outputs["Fac"], mix1.inputs["A"])
    l.new(fine_out, mix1.inputs["B"])
    ramp = node(n, "ShaderNodeValToRGB", (-400, 150))
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.30, 0.72
    ramp.color_ramp.elements[0].color = (0.13, 0.135, 0.14, 1)
    ramp.color_ramp.elements[1].color = (0.34, 0.345, 0.35, 1)
    l.new(mix1.outputs["Result"], ramp.inputs["Fac"])
    # 衬砌环缝：每 1.5 m 一道暗缝
    div = node(n, "ShaderNodeMath", (-1000, -350), operation="DIVIDE")
    div.inputs[1].default_value = 1.5
    l.new(sep.outputs["Y"], div.inputs[0])
    fr = node(n, "ShaderNodeMath", (-850, -350), operation="FRACT")
    l.new(div.outputs[0], fr.inputs[0])
    lt = node(n, "ShaderNodeMath", (-700, -350), operation="LESS_THAN")
    lt.inputs[1].default_value = 0.006
    l.new(fr.outputs[0], lt.inputs[0])
    seam = node(n, "ShaderNodeMix", (-200, 100), data_type="RGBA", blend_type="MULTIPLY")
    l.new(lt.outputs[0], seam.inputs["Factor"])
    l.new(ramp.outputs["Color"], seam.inputs["A"])
    seam.inputs["B"].default_value = (0.25, 0.25, 0.25, 1)
    stain = node(n, "ShaderNodeMix", (0, 150), data_type="RGBA", blend_type="MULTIPLY")
    stain.inputs["Factor"].default_value = 1.0
    l.new(seam.outputs["Result"], stain.inputs["A"])
    l.new(sramp.outputs["Color"], stain.inputs["B"])
    # 下部 0.8 m 偏暗（积灰、返潮）
    zr = node(n, "ShaderNodeMapRange", (-200, -150))
    zr.inputs["From Min"].default_value, zr.inputs["From Max"].default_value = 0.2, 1.2
    zr.inputs["To Min"].default_value, zr.inputs["To Max"].default_value = 0.45, 1.0
    l.new(sep.outputs["Z"], zr.inputs["Value"])
    dirt = node(n, "ShaderNodeMix", (200, 150), data_type="RGBA", blend_type="MULTIPLY")
    dirt.inputs["Factor"].default_value = 1.0
    l.new(stain.outputs["Result"], dirt.inputs["A"])
    l.new(zr.outputs["Result"], dirt.inputs["B"])
    l.new(dirt.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.82
    bump = node(n, "ShaderNodeBump", (300, -300), Strength=0.12, Distance=0.004)
    l.new(fine.outputs["Fac"], bump.inputs["Height"])
    l.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def floor_mat():
    m, n, l = mat_new("floor")
    b = bsdf_of(n)
    nz = node(n, "ShaderNodeTexNoise", (-600, 0), Scale=25.0, Detail=8.0)
    rp = node(n, "ShaderNodeValToRGB", (-400, 0))
    rp.color_ramp.elements[0].color, rp.color_ramp.elements[1].color = (0.03, 0.03, 0.03, 1), (0.13, 0.12, 0.11, 1)
    l.new(nz.outputs["Fac"], rp.inputs["Fac"])
    l.new(rp.outputs["Color"], b.inputs["Base Color"])
    wet = node(n, "ShaderNodeTexNoise", (-600, -300), Scale=1.2, Detail=3.0)
    wr = node(n, "ShaderNodeMapRange", (-400, -300))
    wr.inputs["From Min"].default_value, wr.inputs["From Max"].default_value = 0.45, 0.6
    wr.inputs["To Min"].default_value, wr.inputs["To Max"].default_value = 0.85, 0.18
    l.new(wet.outputs["Fac"], wr.inputs["Value"])
    l.new(wr.outputs["Result"], b.inputs["Roughness"])
    bump = node(n, "ShaderNodeBump", (-200, -500), Strength=0.6, Distance=0.02)
    l.new(nz.outputs["Fac"], bump.inputs["Height"])
    l.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def simple_mat(name, color, rough=0.5, metal=0.0, emit=None, strength=0.0):
    m, n, l = mat_new(name)
    b = bsdf_of(n)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def decal_mat(glow=True):
    """样板墙面贴花 + 裂缝发光（原有琥珀色、新增红色，强度由节点值打关键帧）。"""
    m, n, l = mat_new("decal")
    b = bsdf_of(n)
    b.location = (600, 0)
    tc = node(n, "ShaderNodeTexCoord", (-900, 0))
    t = node(n, "ShaderNodeTexImage", (-600, 200))
    t.image = img("wall_new_marks.png")
    t.interpolation = "Cubic"
    l.new(tc.outputs["UV"], t.inputs["Vector"])
    cr = node(n, "ShaderNodeValToRGB", (-350, 250))
    els = cr.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.0, 0.0, 0.0, 1)
    els[1].position, els[1].color = 1.0, (1.0, 1.0, 1.0, 1)
    for pos, val in ((0.035, 0.012), (0.06, 0.14), (0.19, 0.165), (0.40, 0.19), (0.75, 0.21), (0.92, 0.85)):
        e = els.new(pos); e.color = (val, val, val, 1)
    l.new(t.outputs["Color"], cr.inputs["Fac"])
    tint = node(n, "ShaderNodeMix", (-200, 250), data_type="RGBA", blend_type="MULTIPLY")
    tint.inputs["Factor"].default_value = 1.0
    l.new(cr.outputs["Color"], tint.inputs["A"])
    tint.inputs["B"].default_value = (0.98, 1.0, 1.02, 1)
    l.new(tint.outputs["Result"], b.inputs["Base Color"])
    l.new(t.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = 0.85
    m.surface_render_method = "DITHERED"
    old = node(n, "ShaderNodeTexImage", (-600, -150))
    old.image = img("crack_old.png", True)
    new = node(n, "ShaderNodeTexImage", (-600, -450))
    new.image = img("crack_new.png", True)
    for x in (old, new):
        l.new(tc.outputs["UV"], x.inputs["Vector"])
    g_old = node(n, "ShaderNodeValue", (-600, -700), label="glow_old")
    g_new = node(n, "ShaderNodeValue", (-600, -800), label="glow_new")
    g_old.name, g_new.name = "glow_old", "glow_new"
    g_old.outputs[0].default_value = g_new.outputs[0].default_value = 0.0
    mo = node(n, "ShaderNodeMath", (-300, -150), operation="MULTIPLY")
    l.new(old.outputs["Color"], mo.inputs[0]); l.new(g_old.outputs[0], mo.inputs[1])
    mn = node(n, "ShaderNodeMath", (-300, -450), operation="MULTIPLY")
    l.new(new.outputs["Color"], mn.inputs[0]); l.new(g_new.outputs[0], mn.inputs[1])
    co = node(n, "ShaderNodeMix", (-100, -150), data_type="RGBA")
    co.inputs["A"].default_value = (1.0, 0.62, 0.12, 1)
    co.inputs["B"].default_value = (1.0, 0.08, 0.04, 1)
    tot = node(n, "ShaderNodeMath", (-100, -450), operation="ADD")
    l.new(mo.outputs[0], tot.inputs[0]); l.new(mn.outputs[0], tot.inputs[1])
    frac = node(n, "ShaderNodeMath", (-100, -650), operation="DIVIDE")
    l.new(mn.outputs[0], frac.inputs[0])
    eps = node(n, "ShaderNodeMath", (-250, -650), operation="ADD")
    eps.inputs[1].default_value = 1e-4
    l.new(tot.outputs[0], eps.inputs[0]); l.new(eps.outputs[0], frac.inputs[1])
    l.new(frac.outputs[0], co.inputs["Factor"])
    l.new(co.outputs["Result"], b.inputs["Emission Color"])
    l.new(tot.outputs[0], b.inputs["Emission Strength"])
    return m, g_old, g_new


# ---------------------------------------------------------------- 隧道
def build_tunnel():
    prof = [(-TUN_HALF, 0.0), (-TUN_HALF, WALL_H * 0.5), (-TUN_HALF, WALL_H)]
    for i in range(1, 32):
        a = math.pi - math.pi * i / 32
        prof.append((TUN_HALF * math.cos(a), WALL_H + TUN_HALF * math.sin(a)))
    prof += [(TUN_HALF, WALL_H), (TUN_HALF, WALL_H * 0.5), (TUN_HALF, 0.0)]
    seg = 1.5
    ys = [Y0 + i * seg for i in range(int((Y1 - Y0) / seg) + 1)]
    verts = [(x, y, z) for y in ys for (x, z) in prof]
    P = len(prof)
    faces = []
    for j in range(len(ys) - 1):
        for i in range(P - 1):
            a, b = j * P + i, j * P + i + 1
            faces.append((a, a + P, b + P, b))       # 法线朝内
    ob = mesh_obj("lining", verts, faces, concrete_mat(), smooth=True)
    # 地面（道床）、轨道、轨枕、检修道、电缆槽
    fl = floor_mat()
    mesh_obj("floor", [(-TUN_HALF, Y0, 0), (TUN_HALF, Y0, 0), (TUN_HALF, Y1, 0), (-TUN_HALF, Y1, 0)], [(0, 1, 2, 3)], fl)
    steel = simple_mat("steel", (0.55, 0.55, 0.56), 0.28, 1.0)
    sleeper = simple_mat("sleeper", (0.32, 0.31, 0.3), 0.8)
    for x in (-0.7175, 0.7175):
        box("rail", (x - 0.6, (Y0 + Y1) / 2, 0.24), (0.07, Y1 - Y0, 0.15), steel)
    sl = box("sleeper", (-0.6, Y0 + 0.3, 0.1), (2.5, 0.25, 0.2), sleeper)
    ar = sl.modifiers.new("arr", "ARRAY")
    ar.count, ar.relative_offset_displace = int((Y1 - Y0) / 0.6), (0, 2.4, 0)
    walk = simple_mat("walk", (0.22, 0.215, 0.2), 0.75)
    box("walkway", (3.8, (Y0 + Y1) / 2, 0.25), (1.4, Y1 - Y0, 0.5), walk)
    box("walkway_l", (-3.9, (Y0 + Y1) / 2, 0.25), (1.2, Y1 - Y0, 0.5), walk)
    tray = simple_mat("tray", (0.3, 0.3, 0.31), 0.45, 0.6)
    for x in (-4.38, 4.38):
        box("tray", (x, (Y0 + Y1) / 2, 2.55), (0.22, Y1 - Y0, 0.12), tray)
    cable = simple_mat("cable", (0.02, 0.02, 0.02), 0.5)
    for k, z in enumerate((2.66, 2.70)):
        box("cable", (-4.38 + 0.04 * k, (Y0 + Y1) / 2, z), (0.05, Y1 - Y0, 0.05), cable)
    # 壁灯
    lampmat = simple_mat("lamp", (1, 1, 1), 0.3, emit=(0.86, 0.93, 1.0), strength=40.0)
    for y in range(4, int(Y1), 9):
        for side in (-1, 1):
            if side == 1 and 55 < y < 66:
                continue                                   # 测点附近不放灯，靠检修照明
            x = side * (TUN_HALF - 0.08)
            box("lampbox", (x, y, 3.25), (0.08, 0.9, 0.12), lampmat)
            ld = bpy.data.lights.new("lamp", "POINT")
            ld.energy, ld.color, ld.shadow_soft_size = 90.0, (0.82, 0.9, 1.0), 0.3
            ld.use_shadow = False
            lo = link_obj(bpy.data.objects.new("lamp", ld))
            lo.location = (x - side * 0.25, y, 3.15)
    return ob


def build_patch(glow=True):
    m, g_old, g_new = decal_mat(glow)
    c = PATCH_C
    hw, hh = PATCH_W / 2, PATCH_H / 2
    v = [(c.x, c.y + hw, c.z + hh), (c.x, c.y - hw, c.z + hh), (c.x, c.y - hw, c.z - hh), (c.x, c.y + hw, c.z - hh)]
    me = bpy.data.meshes.new("patch")
    me.from_pydata(v, [], [(0, 3, 2, 1)])
    uv = me.uv_layers.new()
    for li, (uu, vv) in zip(range(4), [(0, 1), (0, 0), (1, 0), (1, 1)]):
        pass
    # 顶点顺序 (0,3,2,1)：左上→左下→右下→右上
    coords = {0: (0, 1), 3: (0, 0), 2: (1, 0), 1: (1, 1)}
    for loop in me.loops:
        uv.data[loop.index].uv = coords[loop.vertex_index]
    ob = link_obj(bpy.data.objects.new("patch", me))
    ob.data.materials.append(m)
    return ob, g_old, g_new


def world(density=0.018, color=(0.6, 0.65, 0.75)):
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    n, l = w.node_tree.nodes, w.node_tree.links
    n["Background"].inputs["Color"].default_value = (0.002, 0.0025, 0.003, 1)
    if density > 0:
        vol = node(n, "ShaderNodeVolumePrincipled", (0, -300), Density=density, Anisotropy=0.55)
        vol.inputs["Color"].default_value = (*color, 1)
        l.new(vol.outputs[0], n["World Output"].inputs["Volume"])


def camera(lens=24, dof=None):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cd.sensor_fit = "AUTO"
    cd.clip_start = 0.02
    co = link_obj(bpy.data.objects.new("cam", cd))
    sc.camera = co
    tgt = link_obj(bpy.data.objects.new("target", None))
    c = co.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    if dof:
        cd.dof.use_dof = True
        cd.dof.focus_object = tgt
        cd.dof.aperture_fstop = dof
    return co, tgt, cd


def kloc(ob, frame, loc):
    ob.location = loc
    ob.keyframe_insert("location", frame=frame)


def phone_model(name, screen_img, emit=1.6):
    """手机：圆角机身 + 屏幕（App 截图作为自发光贴图）。本地坐标：屏幕朝 -Y，长边沿 Z。"""
    body = simple_mat("phone_body", (0.015, 0.016, 0.018), 0.22, 0.4)
    frame = simple_mat("phone_frame", (0.6, 0.6, 0.62), 0.25, 1.0)
    ob = box(name, (0, 0, 0), (0.0745, 0.0082, 0.162), body, bevel=0.0)
    m = ob.modifiers.new("bev", "BEVEL")
    m.width, m.segments, m.limit_method = 0.0095, 10, "NONE"
    rim = box(name + "_rim", (0, 0, 0), (0.0752, 0.0060, 0.1626), frame)
    m2 = rim.modifiers.new("bev", "BEVEL")
    m2.width, m2.segments = 0.0096, 10
    rim.parent = ob
    # 屏幕
    sm, n, l = mat_new(name + "_screen")
    b = bsdf_of(n)
    t = node(n, "ShaderNodeTexImage", (-400, 0))
    t.image = img(screen_img)
    b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
    l.new(t.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = emit
    b.inputs["Roughness"].default_value = 0.08
    b.inputs["Coat Weight"].default_value = 1.0
    sw, sh = 0.0675, 0.150
    v = [(-sw / 2, -0.00415, sh / 2), (sw / 2, -0.00415, sh / 2), (sw / 2, -0.00415, -sh / 2), (-sw / 2, -0.00415, -sh / 2)]
    me = bpy.data.meshes.new(name + "_scr")
    me.from_pydata(v, [], [(0, 3, 2, 1)])
    uv = me.uv_layers.new()
    coords = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
    for loop in me.loops:
        uv.data[loop.index].uv = coords[loop.vertex_index]
    so = link_obj(bpy.data.objects.new(name + "_scr", me))
    so.data.materials.append(sm)
    so.parent = ob
    # 摄像头凸台（背面）
    cam_b = box(name + "_cam", (-0.022, 0.0055, 0.062), (0.026, 0.003, 0.026), body, bevel=0.004)
    cam_b.parent = ob
    return ob, b


def beam(name, p1, p2, r, mat):
    d = Vector(p2) - Vector(p1)
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=r, depth=d.length, location=(Vector(p1) + Vector(p2)) / 2)
    ob = bpy.context.active_object
    ob.name = name
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("Z", "Y")
    ob.data.materials.append(mat)
    return ob


def area_light(name, loc, aim, energy, color, size=0.5, spot=False):
    ld = bpy.data.lights.new(name, "SPOT" if spot else "AREA")
    ld.energy, ld.color = energy, color
    if spot:
        ld.spot_size, ld.spot_blend, ld.shadow_soft_size = math.radians(40), 0.6, 0.05
    else:
        ld.size = size
    lo = link_obj(bpy.data.objects.new(name, ld))
    lo.location = loc
    lo.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return lo, ld


def ease_all():
    for ob in list(bpy.data.objects) + list(bpy.data.lights) + list(bpy.data.materials):
        ad = getattr(ob, "animation_data", None)
        if ob.__class__.__name__ == "Material" and ob.node_tree and ob.node_tree.animation_data:
            ad = ob.node_tree.animation_data
        if not ad or not ad.action:
            continue
        for fc in iter_fcurves(ad.action):
            for k in fc.keyframe_points:
                k.interpolation = "BEZIER"
                k.easing = "AUTO"
                k.handle_left_type = k.handle_right_type = "AUTO_CLAMPED"


def iter_fcurves(action):
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


# ---------------------------------------------------------------- 各镜头
def shot_tunnel():
    """冷开场：沿隧道推进，转向侧壁，停在裂缝前；裂缝先泛琥珀色，随后新增段转红。"""
    build_tunnel()
    patch, g_old, g_new = build_patch()
    world(0.03, (0.75, 0.82, 0.95))
    cam, tgt, cd = camera(24 if ORI == "h" else 20, dof=2.2)
    # 检修灯：随镜头靠近照亮测点
    head, hl = area_light("inspect", (3.6, 59.2, 1.9), CRACK_C, 0.0, (0.85, 0.9, 1.0), size=0.6)
    kf(hl, "energy", 1, 0.0); kf(hl, "energy", 110, 6.0); kf(hl, "energy", 180, 22.0)
    kloc(cam, 1, (0.4, 22.0, 1.75)); kloc(cam, 95, (1.6, 46.0, 1.7)); kloc(cam, 180, (4.12, 59.80, 1.40))
    kloc(tgt, 1, (0.6, 110.0, 2.2)); kloc(tgt, 95, (3.0, 70.0, 1.9)); kloc(tgt, 180, CRACK_C)
    kf(cd, "lens", 1, cd.lens); kf(cd, "lens", 120, cd.lens); kf(cd, "lens", 180, 32 if ORI == "h" else 26)
    kf(cd.dof, "aperture_fstop", 1, 4.0); kf(cd.dof, "aperture_fstop", 180, 2.0)
    go, gn = g_old.outputs[0], g_new.outputs[0]
    kf(go, "default_value", 1, 0.6); kf(go, "default_value", 60, 1.5); kf(go, "default_value", 150, 4.0)
    kf(go, "default_value", 180, 3.0)
    kf(gn, "default_value", 1, 0.0); kf(gn, "default_value", 130, 0.0); kf(gn, "default_value", 165, 9.0)
    kf(gn, "default_value", 180, 6.0)


def shot_end():
    """片尾：从裂缝前缓缓后退、升高，回到隧道中线，灯光一路延伸。"""
    build_tunnel()
    patch, g_old, g_new = build_patch()
    world(0.03, (0.75, 0.82, 0.95))
    cam, tgt, cd = camera(32 if ORI == "h" else 26, dof=3.0)
    head, hl = area_light("inspect", (3.6, 59.2, 1.9), CRACK_C, 18.0, (0.85, 0.9, 1.0), size=0.6)
    kf(hl, "energy", 1, 18.0); kf(hl, "energy", 120, 4.0)
    kloc(cam, 1, (4.12, 59.80, 1.40)); kloc(cam, 90, (2.6, 55.0, 1.8)); kloc(cam, 180, (0.3, 44.0, 2.3))
    kloc(tgt, 1, CRACK_C); kloc(tgt, 90, (4.2, 62.0, 1.6)); kloc(tgt, 180, (0.5, 120.0, 2.6))
    kf(cd, "lens", 1, cd.lens); kf(cd, "lens", 180, 22 if ORI == "h" else 18)
    go, gn = g_old.outputs[0], g_new.outputs[0]
    kf(go, "default_value", 1, 3.0); kf(go, "default_value", 180, 2.0)
    kf(gn, "default_value", 1, 6.0); kf(gn, "default_value", 180, 5.0)


def shot_poses():
    """两次到访：上个月正面、冷光；今天斜着拍、侧面暖光。半透明的手机和取景视锥标出两次的机位。"""
    build_tunnel()
    patch, g_old, g_new = build_patch()
    g_old.outputs[0].default_value = 0.0
    world(0.012)
    cam, tgt, cd = camera(28 if ORI == "h" else 22, dof=None)
    kloc(cam, 1, (2.75, 58.55, 1.68)); kloc(cam, DUR, (2.62, 58.35, 1.72))
    tgt.location = (4.15, 60.2, 1.36)
    cyan = simple_mat("frustumA", (0.2, 0.8, 1.0), 0.5, emit=(0.3, 0.85, 1.0), strength=12.0)
    amber = simple_mat("frustumB", (1.0, 0.6, 0.2), 0.5, emit=(1.0, 0.62, 0.18), strength=12.0)
    corners = [tex_uv(u, v) for u, v in ((0.02, 0.03), (0.98, 0.03), (0.98, 0.97), (0.02, 0.97))]
    out = []
    for tag, pos, look, mat, roll in (("A", (3.70, 60.0, 1.42), PATCH_C, cyan, 0),
                                      ("B", (3.95, 60.62, 1.20), tex_uv(0.45, 0.55), amber, 18)):
        ph, _ = phone_model("phone" + tag, os.path.join(TEX, "scan_screen.png"), emit=0.8)
        ph.location = pos
        q = (Vector(look) - Vector(pos)).to_track_quat("Y", "Z")
        ph.rotation_mode = "QUATERNION"
        ph.rotation_quaternion = q @ Euler((0, math.radians(roll), 0)).to_quaternion()
        lens_pt = Vector(pos)
        parts = [ph] + [beam(f"fr{tag}{i}", lens_pt, c, 0.0022, mat) for i, c in enumerate(corners)]
        parts += [beam(f"ed{tag}{i}", corners[i], corners[(i + 1) % 4], 0.0022, mat) for i in range(4)]
        out.append(parts)
    lA, ldA = area_light("lightA", (3.3, 59.9, 1.6), PATCH_C, 30.0, (0.82, 0.9, 1.0), size=1.2)
    lB, ldB = area_light("lightB", (4.05, 61.3, 1.75), PATCH_C, 0.0, (1.0, 0.62, 0.32), spot=True)
    # 0–50 帧：上个月；50–105：今天
    cut = 52
    for parts, show in ((out[0], (1, cut)), (out[1], (cut, DUR + 1))):
        for p in parts:
            for ob in [p] + list(p.children):
                ob.hide_render = True; ob.keyframe_insert("hide_render", frame=1)
                ob.hide_render = False; ob.keyframe_insert("hide_render", frame=show[0])
                ob.hide_render = True; ob.keyframe_insert("hide_render", frame=show[1])
    kf(ldA, "energy", 1, 30.0); kf(ldA, "energy", cut - 1, 30.0); kf(ldA, "energy", cut, 2.0)
    kf(ldB, "energy", 1, 0.0); kf(ldB, "energy", cut - 1, 0.0); kf(ldB, "energy", cut, 70.0)
    for fc_owner in (ldA, ldB):
        ad = fc_owner.animation_data
        for fc in iter_fcurves(ad.action):
            for k in fc.keyframe_points:
                k.interpolation = "CONSTANT"


def shot_scan():
    """扫码：手机从下方抬起对准二维码，扫描光带掠过二维码，屏幕显示锁定。"""
    build_tunnel()
    patch, g_old, g_new = build_patch()
    world(0.012)
    cam, tgt, cd = camera(30 if ORI == "h" else 26, dof=2.0)
    kloc(cam, 1, (3.80, 59.60, 1.47)); kloc(cam, DUR, (3.84, 59.64, 1.465))
    kloc(tgt, 1, (4.30, 59.98, 1.39)); kloc(tgt, DUR, (4.32, 59.97, 1.39))
    ph, scr = phone_model("phone", os.path.join(TEX, "scan_screen.png"), emit=1.1)
    cd.dof.focus_object = ph
    ph.rotation_mode = "QUATERNION"
    end_pos = Vector((4.05, 59.82, 1.40))
    q = (Vector(QR_C) - end_pos).to_track_quat("Y", "Z")
    q_tilt = q @ Euler((math.radians(-25), 0, math.radians(-8))).to_quaternion()
    ph.rotation_quaternion = q_tilt; ph.keyframe_insert("rotation_quaternion", frame=1)
    ph.rotation_quaternion = q; ph.keyframe_insert("rotation_quaternion", frame=55)
    kloc(ph, 1, end_pos + Vector((-0.10, -0.06, -0.32))); kloc(ph, 55, end_pos); kloc(ph, DUR, end_pos + Vector((0.01, 0, 0.004)))
    # 扫描光带：一条贴着墙面的发光细条，从二维码上方扫到下方
    beammat = simple_mat("scanbeam", (0.1, 1.0, 0.6), 0.5, emit=(0.15, 1.0, 0.55), strength=0.0)
    bs = bsdf_of(beammat.node_tree.nodes)
    sb = box("scanbar", QR_C + Vector((-0.002, 0, 0.04)), (0.002, 0.06, 0.0025), beammat)
    kloc(sb, 60, QR_C + Vector((-0.002, 0, 0.034))); kloc(sb, 100, QR_C + Vector((-0.002, 0, -0.034)))
    es = bs.inputs["Emission Strength"]
    kf(es, "default_value", 58, 0.0); kf(es, "default_value", 62, 40.0); kf(es, "default_value", 98, 40.0)
    kf(es, "default_value", 104, 0.0)
    lA, ldA = area_light("fill", (3.4, 59.6, 1.9), PATCH_C, 14.0, (0.85, 0.9, 1.0), size=1.0)
    rim, ldr = area_light("rim", (4.2, 59.3, 1.9), end_pos, 6.0, (1.0, 0.7, 0.4), size=0.3)


def shot_phones():
    """三部手机展示 App 真实截图，暗色棚拍，缓慢环绕。"""
    world(0.0)
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.007, 1)
    floor = simple_mat("studio", (0.01, 0.012, 0.016), 0.35)
    mesh_obj("studio_floor", [(-5, -5, -0.12), (5, -5, -0.12), (5, 5, -0.12), (-5, 5, -0.12)], [(0, 1, 2, 3)], floor)
    shots = ["截图_对比结论.png", "截图_裂缝标注.png", "截图_系列页.png"]
    if ORI == "h":
        poses = [((-0.105, 0.03, 0), -14), ((0, -0.02, 0.006), 0), ((0.105, 0.03, 0), 14)]
    else:
        poses = [((-0.058, 0.04, 0.035), -10), ((0, -0.03, 0.0), 0), ((0.058, 0.04, -0.035), 10)]
    for i, (s, (p, rz)) in enumerate(zip(shots, poses)):
        ph, _ = phone_model(f"ph{i}", os.path.join(SHOTS_DIR, s), emit=1.0)
        ph.location = p
        ph.rotation_euler = (math.radians(4), 0, math.radians(rz))
        kloc(ph, 1, Vector(p) + Vector((0, 0, -0.012 + 0.006 * i)))
        kloc(ph, DUR, Vector(p) + Vector((0, 0, 0.008 - 0.004 * i)))
    cam, tgt, cd = camera(50 if ORI == "h" else 42, dof=4.0)
    tgt.location = (0, 0, 0.0)
    d = 1.0 if ORI == "h" else 0.78
    kloc(cam, 1, (-0.16 * d, -0.62 * d, 0.06)); kloc(cam, DUR, (0.14 * d, -0.58 * d, 0.02))
    area_light("key", (-0.6, -0.8, 0.8), (0, 0, 0), 60.0, (1.0, 0.93, 0.85), size=1.2)
    area_light("rimL", (-0.5, 0.6, 0.3), (0, 0, 0), 40.0, (1.0, 0.65, 0.2), size=0.4)
    area_light("rimR", (0.5, 0.6, 0.3), (0, 0, 0), 40.0, (0.3, 0.7, 1.0), size=0.4)


dict(tunnel=shot_tunnel, end=shot_end, poses=shot_poses, scan=shot_scan, phones=shot_phones)[SHOT]()
if SHOT != "poses":
    ease_all()

out_dir = os.path.join(HERE, "render", f"{SHOT}_{ORI}")
if STILL:
    sc.frame_set(STILL)
    sc.render.filepath = os.path.join(HERE, "render", f"still_{SHOT}_{ORI}_{STILL}.png")
    bpy.ops.render.render(write_still=True)
else:
    os.makedirs(out_dir, exist_ok=True)
    if FRAMES:
        a, b = FRAMES.split("-")
        sc.frame_start, sc.frame_end = int(a), int(b)
    sc.render.filepath = os.path.join(out_dir, "")
    sc.render.use_overwrite = False
    bpy.ops.render.render(animation=True)
