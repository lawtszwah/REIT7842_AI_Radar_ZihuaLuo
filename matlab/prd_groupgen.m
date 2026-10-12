function prd_groupgen(job_path)
%PRD_GROUPGEN Run the research group's passive-radar generator for a batch of scenes.
%
%   prd_groupgen('job.json')
%
% The group's generator (Dr Lu Zhang, supplied 2026-09-10) is three student interface
% scripts around protected P-code cores. The scripts call `clear` and read and write fixed
% files in the current folder, so this function calls the three cores directly instead,
% with the same arguments the interface scripts pass:
%
%   generate_target_trajectory_core  ->  target trajectory (.mat)
%   Sig_generate_core                ->  reference/surveillance signals r1 [2 x N]
%   RDprocess_core                   ->  range-Doppler cuts, one per frame
%
% The job file is written by prdbench.simulation.groupgen; its fields are documented there.
% Each scene is seeded with rng(seed, 'twister') immediately before the cores run, which
% makes the generated signals bit-identical for a given seed (checked 2026-10-12).
%
% Output, one file per scene: <out_dir>/scene_<id>.mat holding
%   power            single [n_frames x n_doppler x n_range]  |cut|.^2
%   target_position  double [n_frames x 2]                    metres, from the core
%   tausam           double [1 x n_frames]                    bistatic path / (c / fs)
%   fmap, gridx      the processing grids actually used

job = jsondecode(fileread(job_path));
addpath(job.generator_dir);
if ~exist(job.out_dir, 'dir'), mkdir(job.out_dir); end

scenes = job.scenes;
if isstruct(scenes), scenes = num2cell(scenes); end
for k = 1:numel(scenes)
    s = scenes{k};
    work = tempname;
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's'));

    rng(double(s.seed), 'twister');

    traj_file = fullfile(work, 'target_positions_m.mat');
    generate_target_trajectory_core(s.initial_position(:)', s.speed_kmh, s.heading_deg, ...
        s.num_points, s.update_interval, traj_file);
    speed_kmh = s.speed_kmh; heading_deg = s.heading_deg; %#ok<NASGU>
    save(traj_file, 'speed_kmh', 'heading_deg', '-append');
    velocity = s.speed_kmh / 3.6 * [cosd(s.heading_deg), sind(s.heading_deg)];

    cfg = struct();
    cfg.c                = s.c;
    cfg.target_start_idx = s.target_start_idx;
    cfg.range_size       = s.range_size;
    if isfield(s, 'target_free') && s.target_free
        cfg.P_target     = 0;              % target-free map: trajectory kept, no echo
    else
        cfg.P_target     = 10^(s.snr_db / 10);
    end
    % CSR is clutter relative to the target, so CNR = SNR + CSR. A target-free map keeps
    % the clutter level a target at reference_snr_db would have had.
    if isfield(s, 'target_free') && s.target_free
        cfg.P_clutter    = 10^((s.reference_snr_db + s.csr_db) / 10);
    else
        cfg.P_clutter    = 10^((s.snr_db + s.csr_db) / 10);
    end
    cfg.P_direct         = 10^(s.dnr_db / 10);
    cfg.P_noise          = 1;
    cfg.clutter_bins     = s.clutter_bins;
    cfg.decay_factor     = s.decay_factor;
    cfg.min_delay        = s.min_delay;
    cfg.use_random_phase = logical(s.use_random_phase);

    sig_dir = fullfile(work, 'simulated_data');
    Sig_generate_core(s.fs, s.time_duration, s.fc, cfg, s.tx(:)', s.rx(:)', velocity, ...
        traj_file, sig_dir);

    sig = load(fullfile(sig_dir, 'sig1.mat'));
    fmap = s.fmap_start:s.fmap_step:s.fmap_stop;
    gridx = s.gridx_start:1:s.gridx_stop;
    proc = struct('fs', s.fs, 'frame_duration', s.frame_duration, ...
        'frame_step', s.frame_step, 'fmap', fmap, 'gridx', gridx);
    [~, cuts] = RDprocess_core(sig.r1, proc, fullfile(work, 'processed_figures'));

    n_frames = numel(cuts);
    power = zeros(n_frames, numel(fmap), numel(gridx), 'single');
    for f = 1:n_frames
        power(f, :, :) = single(abs(cuts{f}).^2);
    end
    pos = load(fullfile(sig_dir, 'P1.mat'));
    tau = load(fullfile(sig_dir, 'tausam1.mat'));
    target_position = pos.target_position;
    tausam = tau.tausam;

    out = fullfile(job.out_dir, sprintf('scene_%06d.mat', s.id));
    save(out, 'power', 'target_position', 'tausam', 'fmap', 'gridx', '-v7');
    clear cleanup
    fprintf('scene %d/%d done\n', k, numel(scenes));
end
end
