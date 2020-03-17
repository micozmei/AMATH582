% AMATH 582 Final - Apollo 11 Image Blurring/Deblurring
clear all; close all; clc

Global
Color
BW

%% Global Image De-Noising
function Global
    
    % Black/White Image
    bw = imread('moon_bw','bmp');
    bw_d = double(bw);
    bw_t = fft2(bw_d); 

    Width = 0.001; %Filter Width
    kx = 1:size(bw,2); ky = 1:size(bw,1);
    [Kx,Ky] = meshgrid(kx,ky);
    F = exp(-Width*((Kx-size(bw,2)/2).^2+(Ky-size(bw,1)/2).^2));
    Ft = fftshift(F);

    bw_t_f = Ft.*bw_t;
    bw_t = ifft2(bw_t_f);
    bw_f = uint8(abs(bw_t));

    figure(1), 
    imshow(bw), title('Noisy Image','fontsize',20);

    figure(2), 
    imshow(bw_f); title('Filtered Image','fontsize',20);

    figure(3)
    pcolor(log(abs(fftshift(bw_t_f))+1)),shading interp,colormap(hot);
    title('Log Frequency Spectrum','fontsize',20), xlabel('Kx'), ...
        ylabel('Ky');

    % Color Image
    color = imread('moon','jpeg');
    color_d = double(color);
    Fcolor = exp(-Width*((Kx-size(bw,2)/2).^2+(Ky-size(bw,1)/2).^2));
    Fcolor_t = fftshift(Fcolor);
    color_t = zeros(size(bw,1),size(bw,2),3);

    for j=1:3
        color_t(:,:,j) = fft2(color_d(:,:,j));
        colort_f(:,:,j) = Fcolor_t.*color_t(:,:,j);    
        recombined_image(:,:,j) = uint8(abs(ifft2(colort_f(:,:,j))));
    end

    figure(4), pcolor(log(abs(fftshift(color_t(:,:,1)))+1)),...
        shading interp, colormap(hot);
    title('Log Frequency Spectrum','fontsize',20), ...
        xlabel('Kx'), ylabel('Ky');

    figure(5)
    imshow(color), title('Noisy Image','fontsize',20);

    figure(6)
    imshow(recombined_image), title('Filtered Image','fontsize',20);
end


%% Local Blurring (Color)
function Color
    
    % Image is in Color
    color = imread('moon','jpeg');
    color_d = double(color);
    [nx,ny] = size(color_d(:,:,1));

    % Build the Tri-Diagonal Matrix
    x = linspace(0,1,nx); y = linspace(0,1,ny); 
    dx = x(2)-x(1); dy = y(2)-y(1);
    onex = ones(nx,1); oney = ones(ny,1);
    Ix = eye(nx); Iy = eye(ny);
    
    Dx = (spdiags([onex, -2*onex, onex],[-1, 0, 1],nx,nx)/dx^2);
    Dy = (spdiags([oney, -2*oney, oney],[-1, 0, 1],ny,ny)/dy^2);
    
    % Find the Laplacian
    L = kron(Iy,Dx) + kron(Dy,Ix); %Kronecker tensor product

    % Build the Diffusion Coefficients
    D = zeros(nx,ny);
    D(1:220,185:280)=1;
    D = reshape(D,nx*ny,1);

    % Solve the Diffusion Equation
    tspan = [0.00001, 0.00002, 0.0001, 0.001];
    for j=1:3
        An0 = reshape(color_d(:,:,j),nx*ny,1);
        [t, solution] = ode45('img_rhs',tspan,An0,[],L,D);
        figure(j+6)
        for k=1:length(tspan)
            subplot(2,2,k)
            diffused_image(:,:,k) = uint8(reshape(solution(k,:),nx,ny));
            imshow(diffused_image(:,:,k));
            recombined_image(:,:,j,k) = diffused_image(:,:,k);
        end
    end

    figure(10)
    for i=1:length(tspan)
       subplot(2,2,i), imshow(recombined_image(:,:,:,i)), ...
           title(['Diffusion Time: ' num2str(tspan(i)) 's'],'fontsize',20);
    end

    figure(11)
    imshow(color), title('Original Image')
end


%% Local Blurring (Black & White)
function BW
    
    % Image is in Black/White
    bw = imread('moon_bw','bmp');
    bw_d = double(bw);
    [nx,ny] = size(bw_d(:,:));

    % Build the Tri-Diagonal Matrix
    x = linspace(0,1,nx); y = linspace(0,1,ny);
    dx = x(2)-x(1); dy = y(2)-y(1);
    onex = ones(nx,1); oney = ones(ny,1);
    Ix = eye(nx);Iy = eye(ny);
    
    Dx = (spdiags([onex, -2*onex, onex],[-1, 0, 1],nx,nx)/dx^2);
    Dy = (spdiags([oney, -2*oney, oney],[-1, 0, 1],ny,ny)/dy^2);

    % Find the Laplacian
    L = kron(Iy,Dx) + kron(Dy,Ix); %Kronecker tensor product

    % Build the Diffusion Coefficients
    D = zeros(nx,ny);
    D(1:220,185:280)=10;
    D = reshape(D,nx*ny,1);

    % Solve the Diffusion Equation
    tspan = [0.00001, 0.00002, 0.0001, 0.001];
    An0 = reshape(bw_d(:,:),nx*ny,1);
    [t, solution] = ode45('img_rhs',tspan,An0,[],L,D);

    figure(12)
    for k=1:length(tspan)
        subplot(2,2,k)
        diffused_image(:,:,k) = uint8(reshape(solution(k,:),nx,ny));
        imshow(diffused_image(:,:,k)); 
        title(['Diffusion Time: ' num2str(tspan(k)) 's'],'fontsize',20);
    end
end
