const std = @import("std");
const build_options = @import("build_options");
const GLTF = @import("gltf");

// Declaring this makes the parser ignore loading images
pub const BENCHMARK_GLTF = {};

const Command = enum {
    @"--help",
    @"-h",
    @"-r",
};

const DEFAULT_RUNS = 500;

pub fn main() !u8 {
    const allocator = std.heap.smp_allocator;
    const stdout = std.io.getStdOut().writer();
    var args = try std.process.argsWithAllocator(allocator);
    const exe_path = args.next().?;
    var runs: ?u32 = null;
    var asset_path: ?[]const u8 = null;
    while (args.next()) |arg| {
        if (std.meta.stringToEnum(Command, arg)) |cmd| {
            switch (cmd) {
                .@"--help", .@"-h" => {
                    try stdout.print(
                        \\Usage: {s} [OPTION]... [ASSET]
                        \\    -r            Sets number of runs (Default {d})
                        \\    -h, --help    Display this message and exit
                        \\
                    , .{ exe_path, DEFAULT_RUNS });
                    return 0;
                },
                .@"-r" => {
                    if (runs) |_| {
                        try stdout.writeAll("Duplicate argument -r\n");
                        return 1;
                    }
                    const runs_arg = args.next() orelse {
                        try stdout.writeAll("Expected runs number\n");
                        return 1;
                    };
                    runs = std.fmt.parseInt(u32, runs_arg, 10) catch {
                        try stdout.print("Invalid value {s}\n", .{runs_arg});
                        return 1;
                    };
                },
            }
        } else if (asset_path) |_| {
            try stdout.writeAll("Too many arguments\n");
            return 1;
        } else {
            asset_path = arg;
        }
    }

    runs = runs orelse DEFAULT_RUNS;
    asset_path = asset_path orelse {
        try stdout.writeAll("Asset path not passed\n");
        return 1;
    };

    const file = try std.fs.cwd().openFile(asset_path.?, .{});
    defer file.close();
    const data = try file.readToEndAllocOptions(allocator, 100 * 1024 * 1024, null, .of(u8), 0);
    defer allocator.free(data);

    var timer = try std.time.Timer.start();
    var min: u64 = std.math.maxInt(u64);
    var max: u64 = 0;
    var mean: u64 = 0;
    var total: u64 = 0;

    var progress = std.Progress.start(.{ .estimated_total_items = runs.? });
    const runs_node = progress.start("Run", runs.?);
    for (0..runs.?) |_| {
        timer.reset();
        var bm: GLTF.DefaultBufferManager = .empty;
        const dirname = std.fs.path.dirname(asset_path.?) orelse "";
        bm.root_dir = try std.fs.cwd().openDir(dirname, .{});
        const asset = try GLTF.parse(allocator, data, &bm, GLTF.DefaultBufferManager.loadUri, .{});
        defer asset.deinit();
        const took = timer.read();
        min = @min(took, min);
        max = @max(took, max);
        total += took;
        runs_node.completeOne();
    }
    progress.end();

    mean = total / runs.?;

    try stdout.print("min: {}\nmax: {}\nmean: {}\n", .{
        std.fmt.fmtDuration(min),
        std.fmt.fmtDuration(max),
        std.fmt.fmtDuration(mean),
    });

    return 0;
}
