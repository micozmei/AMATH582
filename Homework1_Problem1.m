%Homework #1
clear all; close all; clc;
load Testdata 
L=15; %Spatial domain from -15 to 15
n=64; %Fourier modes from -64 to 63
x2=linspace(-L,L,n+1);
x=x2(1:n);y=x;z=x;
k=(2*pi/(2*L))*[0:(n/2-1),-n/2:-1]; %Wavenumbers
ks=fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
Uave=zeros(n,n,n); %Pre-allocate matrix
for i=1:20 %20 measurements were taken in time
    Un(:,:,:)=reshape(Undata(i,:),n,n,n); %Reshape data file into 64x64x64
    Utn(:,:,:)=fftn(Un(:,:,:)); %To frequency domain
    Uave=Uave+Utn(:,:,:); %Average
    Uaveshift=fftshift(abs(Uave));
    close all, isosurface(Kx,Ky,Kz,Uaveshift/max(abs(Uave(:))),0.75)
    axis([-5 5 -5 5 -5 5])
    xlabel('Kx'),ylabel('Ky'),zlabel('Kz'),title('Frequency Domain') 
    grid on,drawnow,pause(0.05) 
end
[V,I]=max(Uaveshift(:)); %Find max value and index
[kx,ky,kz]=ind2sub(size(Uave),I);
Kxv=Kx(kx,ky,kz);
Kyv=Ky(kx,ky,kz); 
Kzv=Kz(kz,ky,kz); 
width=0.5;
filter=exp(-width*(Kx-Kxv).^2).*exp(-width*(Ky-Kyv).^2)...
.*exp(-width*(Kz-Kzv).^2); %Apply filter
Px=zeros(1,20);Py=Px;Pz=Py;P=zeros(3,20); 
figure()
for j=1:20 %20 measurements were taken in time
    Un(:,:,:)=reshape(Undata(j,:),n,n,n); %Reshape data file into 64x64x64
    Utn(:,:,:)=fftn (Un(:,:,:)); %To frequency domain
    Utnf(:,:,:)=filter.*fftshift(Utn(:,:,:)); %Apply filter
    Unf(:,:,:)=ifftn(Utnf(:,:,:)); %Back to spatial domain
    [V,J]=max(Unf(:)); %Find max value and index
    [Px(j),Py(j),Pz(j)]=ind2sub(size(Unf),J); 
    P(:,j)=[X(Px(j),Py(j),Pz(j));Y(Px(j),Py(j),Pz(j));...
    Z(Px(j),Py(j),Pz(j))]; 
    isosurface(X,Y,Z,abs(Unf)/max(abs(Unf(:))),0.6)
    axis([-20 20 -20 20 -20 20])
    xlabel('X'),ylabel('Y'),zlabel('Z'),title('Spatial Domain')
    grid on,drawnow,pause(0.05)
end 
figure()
plot3(P(1,:),P(2,:),P(3,:),'k-*','Linewidth',[2]) 
axis ([-20 20 -20 20 -20 20]) 
xlabel('X'),ylabel('Y'),zlabel('Z'),title('Spatial Domain'),grid on
fprintf('Marble position at t=0: \n') 
fprintf('X=%d\n',P(1,1)),fprintf('Y=%d\n',P(2,1)),fprintf('Z=%d\n',P(3,1))
fprintf('Marble position at t=20: \n') 
fprintf('X=%d\n',P(1,20)),fprintf('Y=%d\n',P(2,20)),fprintf('Z=%d\n',P(3,20))
