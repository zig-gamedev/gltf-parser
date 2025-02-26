const std = @import("std");
const builtin = @import("builtin");
const stbi = @import("stb_image");
const Allocator = std.mem.Allocator;
const FixedBufferAllocator = std.heap.FixedBufferAllocator;
const base64 = std.base64.url_safe;
const eql = std.mem.eql;
const assert = std.debug.assert;

const GLTF = @This();

fba_buf: []u8,
scene_name: ?[]u8,
scene_nodes: []u16,
nodes: []Node,
meshes: []Mesh,
skins: []Skin,
materials: []Material,
animations: []Animation,

pub const Node = struct {
    mesh: ?u16,
    skin: ?u16,
    children: []u16,
    scale: [3]f32,
    translation: [3]f32,
    rotation: [4]f32,
    weights: [2]f32,
};

pub const Mesh = struct {
    primitives: []Primitive,
};

pub const Skin = struct {
    joints: []u16,
    name: ?[]const u8 = null,
    inverse_bind_matrices: ?u16,
    skeleton: ?u16,
};

pub const Primitive = struct {
    indices: ?[]u32,
    material: ?u16,
    position: ?[]@Vector(3, f32),
    normal: ?[]@Vector(3, f32),
    tangent: ?[]@Vector(4, f32),
    texcoord_0: ?[]@Vector(2, f32),
    texcoord_1: ?[]@Vector(2, f32),
    color_0: ?[]@Vector(4, f32),
    joints_0: ?[]@Vector(4, u16),
    weights_0: ?[]@Vector(4, f32),
};

pub const Material = struct {
    pbr: ?PBR,
    normal: ?Texture,

    const PBR = struct {
        base_color_factor: @Vector(4, f32),
        base_color: ?Texture,
        metallic_roughness: ?Texture,
    };
};

pub const Texture = struct {
    width: u32,
    height: u32,
    data: [:0]u8,
};

pub const Animation = struct {
    name: ?[]const u8,
    channels: []Channel,
    samplers: []Sampler,

    const Channel = struct {
        sampler: u16,
        target: Target,

        const Target = struct {
            node: ?u16,
            path: Path,

            const Path = enum {
                translation,
                rotation,
                scale,
                weights,
            };
        };
    };

    const Sampler = struct {
        input: u16,
        interpolation: Interpolation,
        output: u16,

        const Interpolation = enum {
            linear,
            step,
            cubicspline,
        };
    };
};

const Metadata = struct {
    asset: struct {
        generator: ?[]u8 = null,
        version: []u8,
    },
    scene: u16,
    scenes: []struct {
        name: ?[]u8 = null,
        nodes: []u16,
    },
    nodes: []struct {
        mesh: ?u16 = null,
        skin: ?u16 = null,
        children: []u16 = &.{},
        scale: @Vector(3, f32) = .{ 1, 1, 1 },
        translation: @Vector(3, f32) = .{ 0, 0, 0 },
        rotation: @Vector(4, f32) = .{ 0, 0, 0, 1 },
        weights: @Vector(2, f32) = .{ 0, 0 },
    },
    meshes: []struct {
        primitives: []struct {
            attributes: struct {
                POSITION: ?u16 = null,
                NORMAL: ?u16 = null,
                TANGENT: ?u16 = null,
                TEXCOORD_0: ?u16 = null,
                TEXCOORD_1: ?u16 = null,
                COLOR_0: ?u16 = null,
            },
            indices: ?u16 = null,
            material: ?u16 = null,
        },
    } = &.{},
    skins: []struct {
        joints: []u16,
        name: ?[]u8 = null,
        inverse_bind_matrices: ?u16 = null,
        skeleton: ?u16 = null,
    } = &.{},
    materials: []struct {
        name: ?[]u8 = null,
        pbrMetallicRoughness: ?struct {
            baseColorFactor: @Vector(4, f32) = .{ 1, 1, 1, 1 },
            baseColorTexture: ?struct { index: u16 } = null,
            metallicRoughnessTexture: ?struct { index: u16 } = null,
        } = null,
        normalTexture: ?struct {
            index: u16,
            scale: u16 = 1,
            texcoord: u16 = 0,
        } = null,
    } = &.{},
    textures: []struct {
        sampler: u16,
        source: u16,
    } = &.{},
    images: []struct {
        name: ?[]u8 = null,
        mimeType: []u8,
        bufferView: u16,
    } = &.{},
    animations: []struct {
        name: ?[]u8 = null,
        channels: []struct {
            sampler: u16,
            target: struct {
                node: ?u16 = null,
                path: []u8,
            },
        },
        samplers: []struct {
            input: u16,
            interpolation: ?[]u8 = null,
            output: u16,
        },
    } = &.{},
    accessors: []struct {
        bufferView: u16,
        type: Type,
        componentType: ComponentType,
        count: u32,
        byteOffset: u32 = 0,
        sparse: ?struct {} = null,
        // NOTE: min/max is required for POSITION attribute
        min: ?[]f32 = null,
        max: ?[]f32 = null,
    },
    bufferViews: []struct {
        buffer: u16,
        byteLength: u32,
        byteOffset: u32 = 0,
        byteStride: ?u32 = null,
        target: ?enum(u16) {
            ARRAY_BUFFER = 34962,
            ELEMENT_ARRAY_BUFFER = 34963,
        } = null,
    },
    buffers: []struct {
        byteLength: u32,
        uri: ?[]const u8 = null,
    },

    const Type = enum {
        SCALAR,
        VEC2,
        VEC3,
        VEC4,
        MAT3,
        MAT4,

        fn len(ty: Type) u8 {
            return switch (ty) {
                .SCALAR => 1,
                .VEC2 => 2,
                .VEC3 => 3,
                .VEC4 => 4,
                .MAT3 => 9,
                .MAT4 => 16,
            };
        }

        fn size(ty: Type, component_type: ComponentType) u8 {
            return ty.len() * component_type.size();
        }
    };

    const ComponentType = enum(u16) {
        i8 = 5120,
        u8 = 5121,
        i16 = 5122,
        u16 = 5123,
        u32 = 5125,
        f32 = 5126,

        fn size(component_type: ComponentType) u8 {
            return switch (component_type) {
                .i8, .u8 => 1,
                .i16, .u16 => 2,
                .u32, .f32 => 4,
            };
        }

        fn max(component_type: ComponentType) u32 {
            return switch (component_type) {
                .i8 => std.math.maxInt(i8),
                .u8 => std.math.maxInt(u8),
                .i16 => std.math.maxInt(i16),
                .u16 => std.math.maxInt(u16),
                .u32 => std.math.maxInt(u32),
                .f32 => unreachable,
            };
        }
    };
};

pub const Error = error{
    OutOfMemory,
    AsseetTooSmall,
    NotSupported,
    InvalidAsset,
    LoadingAsset,
};

pub const Options = struct {
    skip_materials: bool = false,
    skip_animations: bool = false,
    skip_skins: bool = false,
};

pub const DefaultBufferManager = struct {
    cwd: ?std.fs.Dir,

    pub const empty: DefaultBufferManager = .{ .cwd = null };

    pub fn loadUri(
        bm: *DefaultBufferManager,
        allocator: Allocator,
        uri: []const u8,
        len: u32,
    ) error{LoadingAsset}![]const u8 {
        if (std.Uri.parse(uri)) |puri| {
            const i = std.mem.indexOf(u8, puri.path.percent_encoded, "base64,") orelse return error.LoadingAsset;
            const base64_data = puri.path.percent_encoded[i + 7 ..];
            const base64_len = base64.Decoder.calcSizeForSlice(base64_data) catch return error.LoadingAsset;
            const data = allocator.alloc(u8, base64_len) catch return error.LoadingAsset;
            base64.Decoder.decode(data, base64_data) catch return error.LoadingAsset;
            if (data.len != len) return error.LoadingAsset;
            return data;
        } else |_| {}

        const cwd = if (bm.cwd) |cwd| cwd else std.fs.cwd();
        const data = cwd.readFileAlloc(allocator, uri, len) catch return error.LoadingAsset;
        return data[0..len];
    }
};

pub fn parse(
    gpa: Allocator,
    data: []const u8,
    ctx: anytype,
    loadUri: *const fn (ctx: @TypeOf(ctx), allocator: Allocator, uri: []const u8, len: u32) error{LoadingAsset}![]const u8,
    options: Options,
) Error!GLTF {
    var fbs = std.io.fixedBufferStream(data);
    const reader = fbs.reader();

    // Header
    const magic = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
    const json_bytes, const is_bin = if (magic == 0x46546C67) blk: {
        // Asset is binary glTF
        const version = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (version != 2) return error.NotSupported;

        const length = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (length <= 12) return error.AsseetTooSmall;

        // JSON Chunk
        var chunk_len = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (chunk_len % 4 != 0) return error.InvalidAsset;

        var chunk_type = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (chunk_type != 0x4E4F534A) return error.InvalidAsset;

        if (fbs.pos + chunk_len > data.len) return error.AsseetTooSmall;
        const json_bytes = data[fbs.pos..][0..chunk_len];
        fbs.pos += chunk_len;

        // Binary Chunk
        chunk_len = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (chunk_len % 4 != 0) return error.InvalidAsset;

        chunk_type = reader.readInt(u32, .little) catch return error.AsseetTooSmall;
        if (chunk_type != 0x004E4942) return error.InvalidAsset;

        break :blk .{ json_bytes, true };
    } else blk: {
        break :blk .{ data, false };
    };

    // Parse JSON
    const parsed_json = std.json.parseFromSlice(
        Metadata,
        gpa,
        json_bytes,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_if_needed },
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidAsset,
    };
    defer parsed_json.deinit();
    const metadata = parsed_json.value;

    // Allocate
    const alloc_size = blk: {
        var alloc_size: usize =
            @sizeOf(Node) * metadata.nodes.len +
            @sizeOf(Skin) * metadata.skins.len +
            @sizeOf(Material) * metadata.materials.len +
            @sizeOf(Animation) * metadata.animations.len;
        for (metadata.buffers) |buf| alloc_size += buf.byteLength;
        for (metadata.meshes) |mesh| alloc_size += @sizeOf(Mesh) + @sizeOf(Primitive) * mesh.primitives.len;
        break :blk alloc_size;
    };
    const fba_buf = try gpa.alloc(u8, alloc_size);
    errdefer gpa.free(fba_buf);

    var fallback_allocator: FallbackAllocator = .{ .fallback = gpa, .fba = .init(fba_buf) };
    const allocator = fallback_allocator.allocator();

    // Parse accessors/images/etc
    const scene = metadata.scenes[metadata.scene];
    const out_nodes = try allocator.alloc(Node, metadata.nodes.len);
    const out_meshes = try allocator.alloc(Mesh, metadata.meshes.len);
    const out_skins = try allocator.alloc(Skin, if (options.skip_skins) 0 else metadata.skins.len);
    const out_materials = try allocator.alloc(Material, if (options.skip_materials) 0 else metadata.materials.len);
    const out_animations = try allocator.alloc(Animation, if (options.skip_animations) 0 else metadata.animations.len);

    for (metadata.nodes, out_nodes) |node, *out_node| {
        out_node.* = .{
            // Even calls to alloc/dupe/free with 0 size has significant time cost
            .children = if (node.children.len > 0) try allocator.dupe(u16, node.children) else &.{},
            .mesh = node.mesh,
            .skin = node.skin,
            .scale = node.scale,
            .translation = node.translation,
            .rotation = node.rotation,
            .weights = node.weights,
        };
    }

    if (!options.skip_materials) {
        for (metadata.materials, out_materials) |material, *out_material| {
            var pbr: ?Material.PBR = null;
            var normal: ?Texture = null;
            if (material.pbrMetallicRoughness) |pbr_metallic_roughness| {
                var base_color: ?Texture = null;

                if (pbr_metallic_roughness.baseColorTexture) |tex_index| {
                    const tex = metadata.textures[tex_index.index];
                    const img = metadata.images[tex.source];
                    const buffer_view = metadata.bufferViews[img.bufferView];
                    if (buffer_view.byteStride != null) return error.InvalidAsset;
                    const buffer = metadata.buffers[buffer_view.buffer];
                    const buffer_data = if (is_bin)
                        data[fbs.pos..][0..buffer.byteLength]
                    else
                        try loadUri(ctx, allocator, buffer.uri.?, buffer.byteLength);
                    const ptr = buffer_data[buffer_view.byteOffset..][0..buffer_view.byteLength];
                    base_color = try loadTexture(ptr);
                }

                var metallic_roughness: ?Texture = null;
                if (pbr_metallic_roughness.metallicRoughnessTexture) |tex_index| {
                    const tex = metadata.textures[tex_index.index];
                    const img = metadata.images[tex.source];
                    const buffer_view = metadata.bufferViews[img.bufferView];
                    if (buffer_view.byteStride != null) return error.InvalidAsset;
                    const buffer = metadata.buffers[buffer_view.buffer];
                    const buffer_data = if (is_bin)
                        data[fbs.pos..][0..buffer.byteLength]
                    else
                        try loadUri(ctx, allocator, buffer.uri.?, buffer.byteLength);
                    const ptr = buffer_data[buffer_view.byteOffset..][0..buffer_view.byteLength];
                    metallic_roughness = try loadTexture(ptr);
                }

                pbr = .{
                    .base_color_factor = pbr_metallic_roughness.baseColorFactor,
                    .base_color = base_color,
                    .metallic_roughness = metallic_roughness,
                };
            }

            if (material.normalTexture) |tex_index| {
                const tex = metadata.textures[tex_index.index];
                const img = metadata.images[tex.source];
                const buffer_view = metadata.bufferViews[img.bufferView];
                if (buffer_view.byteStride != null) return error.InvalidAsset;
                const buffer = metadata.buffers[buffer_view.buffer];
                const buffer_data = if (is_bin)
                    data[fbs.pos..][0..buffer.byteLength]
                else
                    try loadUri(ctx, allocator, buffer.uri.?, buffer.byteLength);
                const ptr = buffer_data[buffer_view.byteOffset..][0..buffer_view.byteLength];
                normal = try loadTexture(ptr);
            }

            out_material.* = .{
                .pbr = pbr,
                .normal = normal,
            };
        }
    }

    if (!options.skip_animations) {
        for (metadata.animations, out_animations) |animation, *out_animation| {
            if (animation.channels.len == 0) return error.InvalidAsset;
            const out_channels = try allocator.alloc(Animation.Channel, animation.channels.len);

            for (animation.channels, out_channels) |channel, *out_channel| {
                const path: Animation.Channel.Target.Path = blk: {
                    if (eql(u8, channel.target.path, "translation")) {
                        break :blk .translation;
                    } else if (eql(u8, channel.target.path, "rotation")) {
                        break :blk .rotation;
                    } else if (eql(u8, channel.target.path, "scale")) {
                        break :blk .scale;
                    } else if (eql(u8, channel.target.path, "weights")) {
                        break :blk .weights;
                    }
                    return error.InvalidAsset;
                };
                out_channel.* = .{
                    .sampler = channel.sampler,
                    .target = .{
                        .node = channel.target.node,
                        .path = path,
                    },
                };
            }

            if (animation.samplers.len == 0) return error.InvalidAsset;
            const out_samplers = try allocator.alloc(Animation.Sampler, animation.samplers.len);

            for (animation.samplers, out_samplers) |sampler, *out_sampler| {
                const interpolation: Animation.Sampler.Interpolation = blk: {
                    if (sampler.interpolation) |intr| {
                        if (eql(u8, intr, "LINEAR")) {
                            break :blk .linear;
                        } else if (eql(u8, intr, "STEP")) {
                            break :blk .step;
                        } else if (eql(u8, intr, "CUBICSPLINE")) {
                            break :blk .cubicspline;
                        }
                        return error.InvalidAsset;
                    }
                    break :blk .linear;
                };

                out_sampler.* = .{
                    .input = sampler.input,
                    .interpolation = interpolation,
                    .output = sampler.output,
                };
            }

            out_animation.* = .{
                .name = animation.name,
                .channels = out_channels,
                .samplers = out_samplers,
            };
        }
    }

    if (!options.skip_skins) {
        for (metadata.skins, out_skins) |skin, *out_skin| {
            if (skin.joints.len == 0) return error.InvalidAsset;

            out_skin.* = .{
                .joints = skin.joints,
                .name = skin.name,
                .inverse_bind_matrices = skin.inverse_bind_matrices,
                .skeleton = skin.skeleton,
            };
        }
    }

    for (metadata.meshes, out_meshes) |mesh, *out_mesh| {
        if (mesh.primitives.len == 0) return error.InvalidAsset;

        const out_primitives = try allocator.alloc(Primitive, mesh.primitives.len);
        for (mesh.primitives, out_primitives) |prim, *out_prim| {
            var indices: ?[]u32 = null;
            var position: ?[]@Vector(3, f32) = null;
            var normal: ?[]@Vector(3, f32) = null;
            var tangent: ?[]@Vector(4, f32) = null;
            var texcoord_0: ?[]@Vector(2, f32) = null;
            var texcoord_1: ?[]@Vector(2, f32) = null;
            var color_0: ?[]@Vector(4, f32) = null;
            var joints_0: ?[]@Vector(4, u16) = null;
            var weights_0: ?[]@Vector(4, f32) = null;

            inline for (&.{
                .{ prim.indices, .index },
                .{ prim.attributes.POSITION, .position },
                .{ prim.attributes.NORMAL, .normal },
                .{ prim.attributes.TANGENT, .tangent },
                .{ prim.attributes.COLOR_0, .color_0 },
                .{ prim.attributes.TEXCOORD_0, .texcoord_0 },
                .{ prim.attributes.TEXCOORD_1, .texcoord_1 },
            }) |entry| if (entry.@"0") |attr| {
                const accessor = metadata.accessors[attr];
                if (accessor.count == 0) return error.InvalidAsset;
                if (accessor.byteOffset % 4 != 0) return error.InvalidAsset;

                const buffer_view = metadata.bufferViews[accessor.bufferView];
                const item_size = accessor.type.size(accessor.componentType);
                const stride = blk: {
                    if (buffer_view.byteStride) |stride| {
                        // Must be aligned to 4-byte
                        if (stride % 4 != 0) return error.InvalidAsset;
                        if (stride < 4 or stride > 252) return error.InvalidAsset;
                        break :blk stride;
                    }
                    // Data is tightly packed
                    break :blk item_size;
                };

                const total_size = stride * (accessor.count - 1) + item_size;
                if (accessor.byteOffset + total_size > buffer_view.byteLength) return error.InvalidAsset;

                const buffer = metadata.buffers[buffer_view.buffer];
                const buffer_data = if (is_bin)
                    data[fbs.pos..][0..buffer.byteLength]
                else
                    try loadUri(ctx, allocator, buffer.uri.?, buffer.byteLength);
                const ptr = buffer_data[buffer_view.byteOffset + accessor.byteOffset ..][0..total_size];

                var iter = std.mem.window(u8, ptr, item_size, stride);
                var i: usize = 0;

                switch (entry.@"1") {
                    .index => {
                        if (accessor.type != .SCALAR) return error.InvalidAsset;

                        indices = try allocator.alloc(u32, accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            indices.?[i] = switch (accessor.componentType) {
                                .u8 => @as(u8, @bitCast(slice[0..@sizeOf(u8)].*)),
                                .u16 => @as(u16, @bitCast(slice[0..@sizeOf(u16)].*)),
                                .u32 => @as(u32, @bitCast(slice[0..@sizeOf(u32)].*)),
                                else => unreachable,
                            };
                            if (indices.?[i] == accessor.componentType.max()) return error.InvalidAsset;
                        }
                    },
                    .position => {
                        if (accessor.type != .VEC3) return error.InvalidAsset;
                        if (accessor.componentType != .f32) return error.InvalidAsset;
                        if (accessor.min == null) return error.InvalidAsset;
                        if (accessor.max == null) return error.InvalidAsset;

                        position = try allocator.alloc(@Vector(3, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            position.?[i] = @bitCast(slice[0..@sizeOf([3]f32)].*);
                        }
                    },
                    .normal => {
                        if (accessor.type != .VEC3) return error.InvalidAsset;
                        if (accessor.componentType != .f32) return error.InvalidAsset;
                        normal = try allocator.alloc(@Vector(3, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            normal.?[i] = @bitCast(slice[0..@sizeOf([3]f32)].*);
                        }
                    },
                    .tangent => {
                        if (accessor.type != .VEC4) return error.InvalidAsset;
                        if (accessor.componentType != .f32) return error.InvalidAsset;
                        tangent = try allocator.alloc(@Vector(4, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            tangent.?[i] = @bitCast(slice[0..@sizeOf([4]f32)].*);
                            if (tangent.?[i][3] < -1 or tangent.?[i][3] > 1) return error.InvalidAsset;
                        }
                    },
                    .texcoord_0 => {
                        if (accessor.type != .VEC2) return error.InvalidAsset;
                        texcoord_0 = try allocator.alloc(@Vector(2, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            texcoord_0.?[i] = switch (accessor.componentType) {
                                .u8 => @floatFromInt(@as(@Vector(2, u8), @bitCast(slice[0..@sizeOf([2]u8)].*))),
                                .u16 => @floatFromInt(@as(@Vector(2, u16), @bitCast(slice[0..@sizeOf([2]u16)].*))),
                                .f32 => @bitCast(slice[0..@sizeOf([2]f32)].*),
                                else => unreachable,
                            };
                        }
                    },
                    .texcoord_1 => {
                        if (accessor.type != .VEC2) return error.InvalidAsset;
                        texcoord_1 = try allocator.alloc(@Vector(2, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            texcoord_1.?[i] = switch (accessor.componentType) {
                                .u8 => @floatFromInt(@as(@Vector(2, u8), @bitCast(slice[0..@sizeOf([2]u8)].*))),
                                .u16 => @floatFromInt(@as(@Vector(2, u16), @bitCast(slice[0..@sizeOf([2]u16)].*))),
                                .f32 => @bitCast(slice[0..@sizeOf([2]f32)].*),
                                else => unreachable,
                            };
                        }
                    },
                    .color_0 => {
                        color_0 = try allocator.alloc(@Vector(4, f32), accessor.count);
                        if (accessor.type == .VEC4) {
                            while (iter.next()) |slice| : (i += 1) {
                                color_0.?[i] = switch (accessor.componentType) {
                                    .u8 => @floatFromInt(@as(@Vector(4, u8), @bitCast(slice[0..@sizeOf([4]u8)].*))),
                                    .u16 => @floatFromInt(@as(@Vector(4, u16), @bitCast(slice[0..@sizeOf([4]u16)].*))),
                                    .f32 => @bitCast(slice[0..@sizeOf([4]f32)].*),
                                    else => unreachable,
                                };
                            }
                        } else if (accessor.type == .VEC3) {
                            while (iter.next()) |slice| : (i += 1) {
                                const color_rgb: @Vector(3, f32) = switch (accessor.componentType) {
                                    .u8 => @floatFromInt(@as(@Vector(3, u8), @bitCast(slice[0..@sizeOf([3]u8)].*))),
                                    .u16 => @floatFromInt(@as(@Vector(3, u16), @bitCast(slice[0..@sizeOf([3]u16)].*))),
                                    .f32 => @bitCast(slice[0..@sizeOf([3]f32)].*),
                                    else => unreachable,
                                };
                                color_0.?[i] = .{ color_rgb[0], color_rgb[1], color_rgb[2], 1 };
                            }
                        } else unreachable;
                    },
                    .joints_0 => {
                        if (accessor.type != .VEC4) return error.InvalidAsset;
                        joints_0 = try allocator.alloc(@Vector(4, u16), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            joints_0.?[i] = switch (accessor.componentType) {
                                .u8 => @as(@Vector(4, u8), @bitCast(slice[0..@sizeOf([4]u8)].*)),
                                .u16 => @as(@Vector(4, u16), @bitCast(slice[0..@sizeOf([4]u16)].*)),
                                else => unreachable,
                            };
                        }
                    },
                    .weights_0 => {
                        if (accessor.type != .VEC4) return error.InvalidAsset;
                        weights_0 = try allocator.alloc(@Vector(4, f32), accessor.count);
                        while (iter.next()) |slice| : (i += 1) {
                            weights_0.?[i] = switch (accessor.componentType) {
                                .u8 => @floatFromInt(@as(@Vector(4, u8), @bitCast(slice[0..@sizeOf([4]u8)].*))),
                                .u16 => @floatFromInt(@as(@Vector(4, u16), @bitCast(slice[0..@sizeOf([4]u16)].*))),
                                .f32 => @bitCast(slice[0..@sizeOf([4]f32)].*),
                                else => unreachable,
                            };
                        }
                    },
                    else => unreachable,
                }
            };

            out_prim.* = .{
                .material = prim.material,
                .indices = indices,
                .position = position,
                .normal = normal,
                .tangent = tangent,
                .texcoord_0 = texcoord_0,
                .texcoord_1 = texcoord_1,
                .color_0 = color_0,
                .joints_0 = joints_0,
                .weights_0 = weights_0,
            };
        }

        out_mesh.* = .{ .primitives = out_primitives };
    }

    return .{
        .fba_buf = fba_buf,
        .scene_name = scene.name,
        .scene_nodes = try allocator.dupe(u16, scene.nodes),
        .nodes = out_nodes,
        .meshes = out_meshes,
        .skins = out_skins,
        .materials = out_materials,
        .animations = out_animations,
    };
}

pub fn deinit(gltf: GLTF, allocator: Allocator) void {
    if (!@hasDecl(@import("root"), "BENCHMARK_GLTF")) {
        for (gltf.materials) |material| {
            if (material.pbr) |pbr| {
                if (pbr.base_color) |tex| stbi.stbi_image_free(tex.data.ptr);
            }
        }
    }
    allocator.free(gltf.fba_buf);
}

fn loadTexture(ptr: []const u8) !Texture {
    var width: c_int = 0;
    var height: c_int = 0;
    var channels_in_file: c_int = 0;
    var image: [:0]u8 = undefined;

    if (!@hasDecl(@import("root"), "BENCHMARK_GLTF")) {
        const image_c = stbi.stbi_load_from_memory(
            ptr.ptr,
            @intCast(ptr.len),
            &width,
            &height,
            &channels_in_file,
            4,
        ) orelse return error.LoadingAsset;
        image = @ptrCast(image_c[0..@intCast(width * height * 4 + 1)]);
    }

    return .{
        .width = @intCast(width),
        .height = @intCast(height),
        .data = image,
    };
}

// TODO: upstream/merge this with std.heap.StackFallbackAllocator
const FallbackAllocator = struct {
    fallback: Allocator,
    fba: FixedBufferAllocator,

    pub fn allocator(fa: *FallbackAllocator) Allocator {
        return .{
            .ptr = fa,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const fa: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        return FixedBufferAllocator.alloc(&fa.fba, len, alignment, ra) orelse
            fa.fallback.rawAlloc(len, alignment, ra);
    }

    fn resize(
        ctx: *anyopaque,
        buf: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ra: usize,
    ) bool {
        const fa: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        if (fa.fba.ownsPtr(buf.ptr)) {
            return FixedBufferAllocator.resize(&fa.fba, buf, alignment, new_len, ra);
        }
        return fa.fallback.rawResize(buf, alignment, new_len, ra);
    }

    fn remap(
        context: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        return_address: usize,
    ) ?[*]u8 {
        const fa: *FallbackAllocator = @ptrCast(@alignCast(context));
        if (fa.fba.ownsPtr(memory.ptr)) {
            return FixedBufferAllocator.remap(&fa.fba, memory, alignment, new_len, return_address);
        }
        return fa.fallback.rawRemap(memory, alignment, new_len, return_address);
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const fa: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        if (fa.fba.ownsPtr(buf.ptr)) {
            return FixedBufferAllocator.free(&fa.fba, buf, alignment, ra);
        }
        return fa.fallback.rawFree(buf, alignment, ra);
    }
};
