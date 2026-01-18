function metrics = calculateMetrics(clean_signal, enhanced_signal, fs)
%CALCULATEMETRICS Compute evaluation metrics for audio quality assessment
%
% Calculates three key metrics for the SP Cup 2026:
%   1. OSINR - Output Signal-to-Interference-plus-Noise Ratio
%   2. ViSQOL - Virtual Speech Quality Objective Listener
%   3. STOI - Short-Time Objective Intelligibility
%
% Inputs:
%   clean_signal    - Reference clean signal [samples x 1]
%   enhanced_signal - Processed/enhanced signal [samples x 1]
%   fs              - Sampling frequency (Hz)
%
% Output:
%   metrics         - Structure containing OSINR, ViSQOL, and STOI scores
%
% Edge Optimization Notes:
%   - Uses single precision where possible
%   - Efficient implementations for real-time feasibility

%% Input validation and preparation
clean_signal = double(clean_signal(:));      % Convert to column vector
enhanced_signal = double(enhanced_signal(:));

% Ensure signals have same length
min_length = min(length(clean_signal), length(enhanced_signal));
clean_signal = clean_signal(1:min_length);
enhanced_signal = enhanced_signal(1:min_length);

% Normalize signals
clean_signal = clean_signal / (max(abs(clean_signal)) + eps);
enhanced_signal = enhanced_signal / (max(abs(enhanced_signal)) + eps);

%% 1. Calculate OSINR (Output Signal-to-Interference-plus-Noise Ratio)
% OSINR measures the ratio of signal power to noise+interference power
% Higher OSINR indicates better separation

% Align signals using cross-correlation (compensate for processing delay)
[correlation, lags] = xcorr(clean_signal, enhanced_signal, 'coeff');
[~, max_idx] = max(abs(correlation));
delay = lags(max_idx);

% Align signals
if delay > 0
    clean_aligned = clean_signal(1:end-delay);
    enhanced_aligned = enhanced_signal(delay+1:end);
elseif delay < 0
    clean_aligned = clean_signal(-delay+1:end);
    enhanced_aligned = enhanced_signal(1:end+delay);
else
    clean_aligned = clean_signal;
    enhanced_aligned = enhanced_signal;
end

% Compute signal and noise powers
signal_power = mean(clean_aligned.^2);
noise_power = mean((enhanced_aligned - clean_aligned).^2);

% Calculate OSINR in dB
if noise_power > eps
    osinr = 10 * log10(signal_power / noise_power);
else
    osinr = 100;  % Very high SNR
end

%% 2. Calculate ViSQOL (Virtual Speech Quality Objective Listener)
% ViSQOL is MATLAB's updated objective speech quality metric
% Score ranges from 1 (bad) to 5 (excellent)

try
    % Check if Audio Toolbox is available
    if license('test', 'Audio_Toolbox')
        visqol_score = visqol(clean_signal, enhanced_signal, fs);
    else
        warning('Audio Toolbox not available. Using PESQ-based approximation for ViSQOL.');
        visqol_score = approximateViSQOL(clean_signal, enhanced_signal, fs);
    end
catch ME
    warning('ViSQOL calculation failed: %s. Using approximation.', ME.message);
    visqol_score = approximateViSQOL(clean_signal, enhanced_signal, fs);
end

%% 3. Calculate STOI (Short-Time Objective Intelligibility)
% STOI measures speech intelligibility
% Score ranges from 0 (unintelligible) to 1 (perfect intelligibility)

try
    % Check if Audio Toolbox is available
    if license('test', 'Audio_Toolbox')
        stoi_score = stoi(clean_signal, enhanced_signal, fs);
    else
        warning('Audio Toolbox not available. Using correlation-based approximation for STOI.');
        stoi_score = approximateSTOI(clean_signal, enhanced_signal);
    end
catch ME
    warning('STOI calculation failed: %s. Using approximation.', ME.message);
    stoi_score = approximateSTOI(clean_signal, enhanced_signal);
end

%% Package results
metrics.osinr = osinr;
metrics.visqol = visqol_score;
metrics.stoi = stoi_score;

end

function visqol_approx = approximateViSQOL(clean_signal, enhanced_signal, fs)
%APPROXIMATEVISQOL Approximate ViSQOL score when Audio Toolbox unavailable
%
% Uses a combination of SNR and spectral similarity to approximate ViSQOL
% This is a simplified version for cases where the toolbox is not available

    % Calculate SNR
    signal_power = mean(clean_signal.^2);
    noise_power = mean((enhanced_signal - clean_signal).^2);
    snr_db = 10 * log10(signal_power / (noise_power + eps));

    % Calculate spectral similarity
    % Compute power spectral densities
    nfft = 512;
    [pxx_clean, ~] = pwelch(clean_signal, hann(nfft), nfft/2, nfft, fs);
    [pxx_enhanced, ~] = pwelch(enhanced_signal, hann(nfft), nfft/2, nfft, fs);

    % Normalize PSDs
    pxx_clean = pxx_clean / sum(pxx_clean);
    pxx_enhanced = pxx_enhanced / sum(pxx_enhanced);

    % Calculate spectral correlation
    spectral_corr = corr(pxx_clean, pxx_enhanced);

    % Map to ViSQOL-like score (1-5 scale)
    % This is an empirical approximation
    snr_component = min(snr_db / 20, 1);  % Normalize SNR contribution
    visqol_approx = 1 + 4 * (0.5 * snr_component + 0.5 * max(spectral_corr, 0));

    % Clamp to valid range
    visqol_approx = max(1, min(5, visqol_approx));
end

function stoi_approx = approximateSTOI(clean_signal, enhanced_signal)
%APPROXIMATESTOI Approximate STOI score when Audio Toolbox unavailable
%
% Uses short-time correlation as a proxy for intelligibility

    % Parameters
    frame_length = 384;  % ~24 ms at 16 kHz
    overlap = frame_length / 2;

    % Frame signals
    num_frames = floor((length(clean_signal) - frame_length) / overlap) + 1;
    correlations = zeros(num_frames, 1);

    for i = 1:num_frames
        start_idx = (i - 1) * overlap + 1;
        end_idx = start_idx + frame_length - 1;

        if end_idx > length(clean_signal)
            break;
        end

        frame_clean = clean_signal(start_idx:end_idx);
        frame_enhanced = enhanced_signal(start_idx:end_idx);

        % Normalize frames
        frame_clean = frame_clean / (norm(frame_clean) + eps);
        frame_enhanced = frame_enhanced / (norm(frame_enhanced) + eps);

        % Compute correlation
        correlations(i) = max(0, frame_clean' * frame_enhanced);
    end

    % Average correlation as STOI approximation
    stoi_approx = mean(correlations);

    % Clamp to valid range [0, 1]
    stoi_approx = max(0, min(1, stoi_approx));
end
