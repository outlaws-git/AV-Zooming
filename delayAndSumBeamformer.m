function output_signal = delayAndSumBeamformer(mic_signals, target_azimuth, mic_spacing, c, fs)
%DELAYANDSUMBEAMFORMER Delay-and-sum beamformer optimized for edge devices
%
% Inputs:
%   mic_signals     - Microphone signals [samples x num_mics]
%   target_azimuth  - Target direction in degrees
%   mic_spacing     - Distance between microphones (m)
%   c               - Speed of sound (m/s)
%   fs              - Sampling frequency (Hz)
%
% Output:
%   output_signal   - Beamformed signal [samples x 1]
%
% Edge Optimization:
%   - Uses single precision for memory efficiency
%   - Minimal memory allocation
%   - MATLAB Coder compatible
%   - Fixed-point ready algorithm structure

%% Type conversion for edge deployment
mic_signals = single(mic_signals);
target_azimuth = single(target_azimuth);
mic_spacing = single(mic_spacing);
c = single(c);
fs = single(fs);

%% Calculate steering delays
% Convert azimuth to radians (relative to broadside)
theta = deg2rad(90 - target_azimuth);  % Convert to angle from array axis

% Calculate time delay between adjacent microphones
% For a 2-element ULA: tau = (d * sin(theta)) / c
tau = (mic_spacing * sin(theta)) / c;
delay_samples = tau * fs;

%% Apply delays and sum
[num_samples, num_mics] = size(mic_signals);
output_signal = zeros(num_samples, 1, 'single');

if num_mics == 2
    % Optimized for 2-microphone array
    % Apply fractional delay using linear interpolation (efficient for edge)
    if abs(delay_samples) < num_samples
        if delay_samples > 0
            % Delay second microphone
            delayed_signal = fractionalDelay(mic_signals(:, 2), delay_samples);
            output_signal = (mic_signals(:, 1) + delayed_signal) / 2;
        else
            % Delay first microphone
            delayed_signal = fractionalDelay(mic_signals(:, 1), -delay_samples);
            output_signal = (delayed_signal + mic_signals(:, 2)) / 2;
        end
    else
        % If delay is too large, just average
        output_signal = mean(mic_signals, 2);
    end
else
    % General case for N microphones
    for m = 1:num_mics
        delay_m = (m - 1) * delay_samples;
        if abs(delay_m) < num_samples
            delayed_signal = fractionalDelay(mic_signals(:, m), delay_m);
            output_signal = output_signal + delayed_signal;
        end
    end
    output_signal = output_signal / num_mics;
end

% Normalize output
output_signal = output_signal / max(abs(output_signal) + eps('single'));

end

function delayed_signal = fractionalDelay(signal, delay_samples)
%FRACTIONALDELAY Apply fractional sample delay using linear interpolation
%
% This implementation is optimized for edge devices:
%   - Uses simple linear interpolation (low complexity)
%   - Single precision arithmetic
%   - Minimal memory allocation

    signal = single(signal);
    delay_samples = single(delay_samples);

    num_samples = length(signal);
    delayed_signal = zeros(size(signal), 'single');

    % Split into integer and fractional parts
    delay_int = floor(delay_samples);
    delay_frac = delay_samples - delay_int;

    % Apply integer delay
    if delay_int > 0 && delay_int < num_samples
        % Forward shift
        delayed_signal(delay_int+1:end) = signal(1:end-delay_int);

        % Apply fractional delay using linear interpolation
        if delay_frac > 0
            for i = delay_int+2:num_samples
                delayed_signal(i) = (1 - delay_frac) * delayed_signal(i) + ...
                                    delay_frac * delayed_signal(i-1);
            end
        end
    elseif delay_int < 0 && abs(delay_int) < num_samples
        % Backward shift
        delay_int = abs(delay_int);
        delayed_signal(1:end-delay_int) = signal(delay_int+1:end);

        % Apply fractional delay
        if delay_frac > 0
            for i = 1:num_samples-delay_int-1
                delayed_signal(i) = (1 - delay_frac) * delayed_signal(i) + ...
                                    delay_frac * delayed_signal(i+1);
            end
        end
    else
        % Delay is too large or zero
        delayed_signal = signal;
    end
end
