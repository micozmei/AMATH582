% Global Image De-Noising
clear all; close all; clc

% Black/White image
d2 = imread('moon_bw','bmp');
d2d = double(d2);
d2t= fft2(d2d); 

width = 0.001; %Filter Width
kx = 1:size(d2,2); ky = 1:size(d2,1);
[Kx,Ky] = meshgrid(kx,ky);
F = exp(-width*((Kx-size(d2,2)/2).^2+(Ky-size(d2,1)/2).^2));
Ft = fftshift(F);

figure(1), surfl(Kx,Ky,F), shading interp; %Filter visualization
xlabel('X-dir'), ylabel('Y-Dir'), zlabel('Filter Magnitude');
title('Image Layer Filter','fontsize',20), xlim([1, 460]), ylim([1, 288]); %Resolution

d2tf = Ft.*d2t;
d2t = ifft2(d2tf);
d2f = uint8(abs(d2t));

figure(2), 
imshow(d2), title('Noisy Image','fontsize',20);

figure(3), 
imshow(d2f); title('Filtered Image','fontsize',20);

figure(4)
pcolor(log(abs(fftshift(d2tf))+1)),shading interp,colormap(hot);
title('Log Frequency Spectrum','fontsize',20), xlabel('Kx'), ylabel('Ky');

% Color image
dcolor = imread('moon','jpeg');
dcolord = double(dcolor);
width = 0.001;
Fcolor = exp(-width*((Kx-size(d2,2)/2).^2+(Ky-size(d2,1)/2).^2));
Fcolort = fftshift(Fcolor);

dcolort = zeros(size(d2,1),size(d2,2),3);

for j=1:3
    dcolort(:,:,j) = fft2(dcolord(:,:,j));
    dcolortf(:,:,j) = Fcolort.*dcolort(:,:,j);    
    recombined(:,:,j) = uint8(abs(ifft2(dcolortf(:,:,j))));
end

figure(5), pcolor(log(abs(fftshift(dcolort(:,:,1)))+1)),shading interp,colormap(hot);
title('Log Frequency Spectrum','fontsize',20), xlabel('Kx'), ylabel('Ky');

figure(6)
imshow(dcolor), title('Noisy Image','fontsize',20);

figure(7)
imshow(recombined), title('Filtered Image','fontsize',20);
