const std = @import("std");
const builtin = @import("builtin");
const Build = std.Build;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    const build_options_module = build_options.createModule();
    const stb_dep = b.dependency("stb", .{});

    const test_cmd = b.step("test", "Run test");
    const benchmark_cmd = b.step("benchmark", "Run benchmark");

    // STB
    const stb_image_lib = b.addStaticLibrary(.{
        .name = "stb_image",
        .target = target,
        .optimize = .ReleaseFast,
    });
    stb_image_lib.linkLibC();
    stb_image_lib.addIncludePath(stb_dep.path(""));
    stb_image_lib.addCSourceFile(.{ .file = b.path("src/stb_image.c") });

    // STB Image Translate-C
    const stb_image_translate_c = b.addTranslateC(.{
        .root_source_file = stb_dep.path("stb_image.h"),
        .target = target,
        .optimize = optimize,
    });

    // Module
    const module = b.addModule("gltf", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    module.addImport("build_options", build_options_module);
    module.addImport("stb_image", stb_image_translate_c.createModule());
    module.linkLibrary(stb_image_lib);

    // Test
    const test_step = b.addTest(.{
        .root_module = module,
        .target = target,
        .optimize = optimize,
    });
    const run_test = b.addRunArtifact(test_step);
    test_cmd.dependOn(&run_test.step);

    // Benchmark
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    benchmark.root_module.addImport("gltf", module);
    benchmark.root_module.addImport("build_options", build_options_module);
    benchmark.linkLibCpp();
    benchmark.addIncludePath(stb_dep.path(""));
    benchmark.addCSourceFile(.{ .file = b.path("src/stb_image.c") });

    const install_benchmark_exe_cmd = b.addInstallArtifact(benchmark, .{});
    benchmark_cmd.dependOn(&install_benchmark_exe_cmd.step);

    const run_benchmark = b.addRunArtifact(benchmark);
    run_benchmark.addArgs(b.args orelse &.{});
    benchmark_cmd.dependOn(&run_benchmark.step);
}
