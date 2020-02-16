%% Main Test File for PCA Study
clear all, close all, clc;

cam1struct = load('cam1_1.mat'); cam1 = cam1struct.vidFrames1_1;
cam2struct = load('cam2_1.mat'); cam2 = cam2struct.vidFrames2_1;
cam3struct = load('cam3_1.mat'); cam3 = cam3struct.vidFrames3_1;

% cam1struct = load('cam1_2.mat'); cam1 = cam1struct.vidFrames1_2;
% cam2struct = load('cam2_2.mat'); cam2 = cam2struct.vidFrames2_2;
% cam3struct = load('cam3_2.mat'); cam3 = cam3struct.vidFrames3_2;

% cam1struct = load('cam1_3.mat'); cam1 = cam1struct.vidFrames1_3;
% cam2struct = load('cam2_3.mat'); cam2 = cam2struct.vidFrames2_3;
% cam3struct = load('cam3_3.mat'); cam3 = cam3struct.vidFrames3_3;

% cam1struct = load('cam1_4.mat'); cam1 = cam1struct.vidFrames1_4;
% cam2struct = load('cam2_4.mat'); cam2 = cam2struct.vidFrames2_4;
% cam3struct = load('cam3_4.mat'); cam3 = cam3struct.vidFrames3_4;

width=0.0001; n=2; % filter parameters
[track_x_1, track_y_1] = Track_Object(cam1,width,n);
[track_x_1, track_y_1] = InPhase(track_x_1,track_y_1);

[track_x_2, track_y_2] = Track_Object(cam2,width,n);
[track_x_2, track_y_2] = InPhase(track_x_2,track_y_2);

[track_x_3, track_y_3] = Track_Object(cam3,width,n);
[track_x_3, track_y_3] = InPhase(track_x_3,track_y_3);

% Trim to equal length
len = [length(track_x_1);
       length(track_y_1);
       length(track_x_2);
       length(track_y_2);
       length(track_x_3);
       length(track_y_3)];
len_min = min(len);
x1 = track_x_1(1:len_min);y1 = track_y_1(1:len_min);
x2 = track_x_2(1:len_min);y2 = track_y_2(1:len_min);
x3 = track_x_3(1:len_min);y3 = track_y_3(1:len_min);

% Plot Camera Signals
figure(1);
subplot(3,1,1)
hold on;
plot(1:len_min,x1,'r.')
plot(1:len_min,y1,'k.')
title('Cam 1')
xlabel('Frame'), ylabel('Pixel')
axis([0, len_min, 0, 500]), legend('X','Y')

subplot(3,1,2)
hold on;
plot(1:len_min,x2,'r.')
plot(1:len_min,y2,'k.')
title('Cam 2')
xlabel('Frame'), ylabel('Pixel')
axis([0, len_min, 0, 500]), legend('X','Y')

subplot(3,1,3)
hold on;
plot(1:len_min,x3,'r.')
plot(1:len_min,y3,'k.')
title('Cam 3')
xlabel('Frame'), ylabel('Pixel')
axis([0, len_min, 0, 500]), legend('X','Y')


%% Principal Component Analysis
% "A" Matrix
x1m = mean(x1); y1m = mean(y1);
x2m = mean(x2); y2m = mean(y2);
x3m = mean(x3); y3m = mean(y3);

A = [x1-x1m;
     y1-y1m;
     x2-x2m;
     y2-y2m;
     x3-x3m;
     y3-y3m];

 Y = A'./sqrt(len_min-1);

% Apply SVD
[u, s, v] = svd(Y);
V = diag(s).*diag(s);

% Plot singular values
figure(2)
plot(V/sum(V),'ko','linewidth',2)
title('Relative Modal Energy Content')
xlabel('Singular Value'), ylabel('Fractional Value')
axis([1, 6, 0, 1]), grid on, axis square
ax=gca; ax.XTick = [1, 2, 3, 4, 5, 6];

% Plot Weighted Modes
t = 1:len_min;
figure(3)
subplot(3,1,1)
mode1 = u(:,1).*s(1,1);
plot(t,mode1,'k')
axis([0, len_min, -10, 10])
title('Mode 1')
xlabel('Frame'), ylabel('Weighted Amp')

subplot(3,1,2)
mode2 = u(:,2).*s(2,2);
plot(t,mode2,'k')
axis([0, len_min, -10, 10])
title('Mode 2')
xlabel('Frame'), ylabel('Weighted Amp')

subplot(3,1,3)
mode3 = u(:,3).*s(3,3);
plot(t,mode3,'k')
axis([0, len_min, -10, 10])
title('Mode 3')
xlabel('Frame'), ylabel('Weighted Amp')

% 3D Plot Reconstruction
figure(4)
plot3(t,mode2,mode1,'r');
hold on;
plot3(t,mode2,-10+0.*mode1,'k');
plot3(t,10+0.*mode2,mode1,'k');
axis([0, len_min, -10, 10, -10, 10])
grid on;
xlabel('Frames'), ylabel('X Amplitude'), zlabel('Y Amplitude')
title('Two Mode Dimensionless Spatial Reconstruction')


%% Make Videos be in Phase
function [track_1, track_2] = InPhase(track_1, track_2)

    % Find which cam varies more and truncate
    phase1 = track_1; min1=min(phase1); max1=max(phase1); d1=max1-min1;
    phase2 = track_2; min2=min(phase2); max2=max(phase2); d2=max2-min2;

    if (d1 > d2)
        phase = phase1;
        mark1 = true(1);
        mark2 = false(1);
    else
        phase = phase2;
        mark1 = false(1);
        mark2 = true(1);
    end

    for j = 4:length(phase)
        first = phase(j-3);
        second = phase(j-2);
        third = phase(j-1);
        fourth = phase(j);
    
        if (first < second) && (second < third) && (third < fourth)
            phase = phase((j-3):end);
            break
        end
    end

    if mark1
        track_1 = phase;
        track_2 = track_2((j-3):end);
    elseif mark2
        track_2 = phase;
        track_1 = track_1((j-3):end);
    end
end


%% Object Tracking
function [Track_X, Track_Y] = Track_Object(cam,width,n)
    num_Frames = length(cam(1,1,1,:));

    % RGB to Grayscale
    for i = 1:num_Frames
        cam_gray(:,:,i) = rgb2gray(cam(:,:,:,i));
    end

    % Meshgrid
    x = 1:length(cam_gray(1,:,1));
    y = 1:length(cam_gray(:,1,1));
    [X,Y] = meshgrid(x,y);

    % Fourier Transform
    sum = zeros(480,640);
    for j = 1:num_Frames
        cam_gray_t(:,:,j) = fft2(cam_gray(:,:,j));
        sum = sum + cam_gray_t(:,:,j);
    end

    average = sum./j;
    A = reshape(fftshift(average),1,(640*480));
    [M,I] = max(A);
    [x_index, y_index] = ind2sub([480, 640],I);
    x_loc(1)=X(x_index,y_index); y_loc(1)=Y(x_index,y_index);

    % Apply Filter
    for k = 1:num_Frames
        if k==1 % First Frame
            F = exp(-width*((X-x_loc(k)).^n+(Y-y_loc(k)).^n));
        else
            F = exp(-width*((X-x_loc(k)-(x_loc(k)-x_loc(k-1))/2).^n+(Y-y_loc(k)-(y_loc(k)-y_loc(k-1))/2).^n));
        end
        cam_filter(:,:,k) = F.*double(cam_gray(:,:,k));

        % Max finds and centers location of object
        A = reshape(cam_filter(:,:,k),1,(640*480));
        [M,I] = max(A);
        [x_index, y_index] = ind2sub([480, 640],I);
        x_loc(k+1) = X(x_index,y_index);
        y_loc(k+1) = Y(x_index,y_index);
        Track_X(k) = x_loc(k);
        Track_Y(k) = y_loc(k);
    end
end
