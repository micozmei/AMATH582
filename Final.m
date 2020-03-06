% AMATH 582 Final - Apollo 11 Image Blurring/Deblurring
clear all; close all; clc

Global
Color
BW

%% Global Image De-Noising
function Global
% Black/White image
d2 = imread('moon_bw','bmp');
d2d = double(d2);
d2t= fft2(d2d); 

width = 0.001; %Filter Width
kx = 1:size(d2,2); ky = 1:size(d2,1);
[Kx,Ky] = meshgrid(kx,ky);
F = exp(-width*((Kx-size(d2,2)/2).^2+(Ky-size(d2,1)/2).^2));
Ft = fftshift(F);

% figure(1), surfl(Kx,Ky,F), shading interp; %Filter visualization
% xlabel('X-dir'), ylabel('Y-Dir'), zlabel('Filter Magnitude');
% title('Image Layer Filter','fontsize',20), xlim([1, 460]), ylim([1, 288]); %Resolution

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
end


%% Local Blurring (Color)
function Color
% Image is Color
color = imread('moon','jpeg');
colord = double(color);
[nx,ny] = size(colord(:,:,1));

% Build Tri-Diagonal Matrix
x = linspace(0,1,nx); 
y = linspace(0,1,ny); 
dx=x(2)-x(1); 
dy=y(2)-y(1);
onex = ones(nx,1); 
oney = ones(ny,1);

Dx = (spdiags([onex -2*onex onex],[-1 0 1],nx,nx)/dx^2);
Ix = eye(nx);

Dy = (spdiags([oney -2*oney oney],[-1 0 1],ny,ny)/dy^2);
Iy = eye(ny);

% Laplacian
L = kron(Iy,Dx) + kron(Dy,Ix); %Kronecker tensor product

% Build Diffusion Coefficients
D = zeros(nx,ny);
D(1:220,185:280)=1;
D = reshape(D,nx*ny,1);

% Solve Diffusion Equation
tspan = [0.00001, 0.00002, 0.0001, 0.001];
for j=1:3
    An0 = reshape(colord(:,:,j),nx*ny,1);
    [t, solution] = ode45('img_rhs',tspan,An0,[],L,D);
    figure(j+7)
    for k=1:length(tspan)
        subplot(2,2,k)
        diffused(:,:,k) = uint8(reshape(solution(k,:),nx,ny));
        imshow(diffused(:,:,k));
        recombined(:,:,j,k) = diffused(:,:,k);
    end
end

figure(11)
for j=1:length(tspan)
   subplot(2,2,j), imshow(recombined(:,:,:,j)), title(['Diffusion Time:  ' num2str(tspan(j)) 's'],'fontsize',20);
end

figure(12)
imshow(color), title('Original Image')
end


%% Local Blurring (Black & White)
function BW
% Image is Black/White
bw = imread('moon_bw','bmp');
bwd = double(bw);
[nx,ny] = size(bwd(:,:));

% Build Tri-Diagonal Matrix
x = linspace(0,1,nx);
y = linspace(0,1,ny);
dx=x(2)-x(1);
dy=y(2)-y(1);
onex = ones(nx,1);
oney = ones(ny,1);

Dx = (spdiags([onex -2*onex onex],[-1 0 1],nx,nx)/dx^2);
Ix = eye(nx);

Dy = (spdiags([oney -2*oney oney],[-1 0 1],ny,ny)/dy^2);
Iy = eye(ny);

% Laplacian
L = kron(Iy,Dx) + kron(Dy,Ix); %Kronecker tensor product

% Build Diffusion Coefficients
D = zeros(nx,ny);
D(1:220,185:280)=10;
D = reshape(D,nx*ny,1);

% Solve Diffusion Equation
tspan = [0.00001, 0.00002, 0.0001, 0.001];
An0 = reshape(bwd(:,:),nx*ny,1);
[t, solution] = ode45('img_rhs',tspan,An0,[],L,D);

figure()
for k=1:length(tspan)
    subplot(2,2,k)
    diffused(:,:,k) = uint8(reshape(solution(k,:),nx,ny));
    imshow(diffused(:,:,k)); title(['Diffusion Time:  ' num2str(tspan(k)) 's'],'fontsize',20);
end
end
