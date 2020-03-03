% Local Blurring (Color)
clear all; close all; clc

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
    figure(j)
    for k=1:length(tspan)
        subplot(2,2,k)
        diffused(:,:,k) = uint8(reshape(solution(k,:),nx,ny));
        imshow(diffused(:,:,k));
        recombined(:,:,j,k) = diffused(:,:,k);
    end
end

figure(4)
for j=1:length(tspan)
   subplot(2,2,j), imshow(recombined(:,:,:,j)), title(['Diffusion Time:  ' num2str(tspan(j)) 's'],'fontsize',20);
end

figure(5)
imshow(color), title('Original Image')
