% Mircea Cozmei - AMATH 582 HW#3
clear all; close all; clc

Test1 
Test2 
Test3 
Test4 
 
%% Test 1: The Ideal Case 
function Test1 
% Load Camera 1 
load cam1_1.mat
v11 = vidFrames1_1(175:425,225:450,:,:); 
v11 = v11(:,:,:,70:end);
v11size = size(v11,4); 
x11 = zeros(v11size,1);
y11 = x11; 
for i = 1:v11size 
    v11g = rgb2gray(v11(:,:,:,i)); 
    focus = 25; 
    if i == 1 
        x11p = 100; 
        y11p = 160; 
    else 
        y11p = y11(i-1); 
        x11p = x11(i-1); 
    end
    v11g(floor(y11p+focus):end,:)=0;
    v11g(1:floor(y11p-focus),:)=0;
	v11g(:,floor(x11p+focus):end)=0; 
	v11g(:,1:floor(x11p-focus))=0;
	[row,col] = find(v11g == max(max(v11g))); 
	x11(i)= median(col); 
	y11(i)= median(row); 
end 
clearvars -except x11 y11 

% figure()
% subplot(1,3,1)
% plot(x11,'-r','Linewidth',2), hold on 
% plot(y11,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

% Load Camera 2 
load cam2_1.mat
v21 = vidFrames2_1(75:400,200:400,:,:); 
clear vidFrames2_1 
v21 = v21(:,:,:,39:end); 
v21size = size(v21,4); 
x21 = zeros(v21size,1); 
y21 = x21; 
for i = 1:v21size 
    v21g = rgb2gray(v21(:,:,:,i)); 
    focus =25; 
    if i == 1 
        x21p = 45; 
        y21p = 168;
    else 
        y21p = y21(i-1); 
        x21p = x21(i-1); 
    end 
    v21g(floor(y21p+focus):end,:)=0; 
    v21g(1:floor(y21p-focus),:)=0; 
    v21g(:,floor(x21p+focus):end)=0; 
    v21g(:,1:floor(x21p-focus))=0; 
    [row,col] = find(v21g == max(max(v21g))); 
    x21(i)= median(col); 
    y21(i)= median(row); 
end 
clearvars -except x11 y11 x21 y21 

% subplot(1,3,2)
% plot(x21,'-r','Linewidth',2), hold on 
% plot(y21,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 2'), legend('X','Y'), xlabel('Frames')

% Load Camera 3
load cam3_1.mat 
v31 = vidFrames3_1(200:375,225:500,:,:);
v31 = v31(:,:,:,70:end);
clear vidFrames3_1 
v31size = size(v31,4); 
x31 = zeros(v31size,1); 
y31 = x31; 
for i = 1:v31size
    v31g = rgb2gray(v31(:,:,:,i)); 
	focus = 15; 
    if i == 1 
    	x31p = 164; 
    	y31p = 54; 
    else 
        y31p = y31(i-1); 
        x31p = x31(i-1); 
    end
    v31g(floor(y31p+focus):end,:)=0; 
	v31g(1:floor(y31p-focus),:)=0; 
	v31g(:,floor(x31p+focus):end)=0; 
	v31g(:,1:floor(x31p-focus))=0; 
	[row,col] = find(v31g == max(max(v31g))); 
	x31(i)= median(col); 
	y31(i)= median(row);  
end 
clearvars -except x11 y11 x21 y21 x31 y31 

% subplot(1,3,3)
% plot(x31,'-r','Linewidth',2), hold on 
% plot(y31,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 3'), legend('X','Y')

% Reshape Data 
minlen = min([size(x11,1),size(x21,1),size(x31,1)]);
X1 = [x11(1:minlen,:),y11(1:minlen,:),x21(1:minlen,:),... 
      y21(1:minlen,:),x31(1:minlen,:),y31(1:minlen,:)];
[m,n] = size(X1); 
X1mean = mean(X1,1); 
X1m = X1 - repmat(X1mean,m,1); 
X1mn = [X1m(:,1)/(sqrt(max(X1m(:,1))^2+max(X1m(:,2))^2)),... 
        X1m(:,2)/(sqrt(max(X1m(:,1))^2+max(X1m(:,2))^2)),... 
        X1m(:,3)/(sqrt(max(X1m(:,3))^2+max(X1m(:,4))^2)),... 
    	X1m(:,4)/(sqrt(max(X1m(:,3))^2+max(X1m(:,4))^2)),... 
    	X1m(:,5)/(sqrt(max(X1m(:,5))^2+max(X1m(:,6))^2)),... 
    	X1m(:,6)/(sqrt(max(X1m(:,5))^2+max(X1m(:,6))^2))]; 
 
figure()
subplot(1,3,1)
plot(X1mn(:,1),'-r','Linewidth',2), hold on 
plot(X1mn(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 1'), legend('X','Y'), ylabel('Normalized Pixel Location')

subplot(1,3,2)
plot(X1mn(:,3),'-r','Linewidth',2), hold on 
plot(X1mn(:,4),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 2'), legend('X','Y'), xlabel('Frames')

subplot(1,3,3)
plot(X1mn(:,5),'-r','Linewidth',2), hold on 
plot(X1mn(:,6),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 3'), legend('X','Y')

% Eignevalue & Eigenvector 
Cx1=(1/(n-1))*(X1mn*X1mn'); % covariance 
[V1,D1]=eig(Cx1); % eigenvectors(V)/eigenvalues(D)
lambda1e=diag(D1); % get eigenvalues 
[dummy,m_arrange1]=sort(-1*abs(lambda1e)); % sort in decreasing order 
lambda1e=lambda1e(m_arrange1); 

% SVD 
[u1,s1,v1] = svd(X1mn/sqrt(n-1),'econ'); % perform the SVD 
Y1s = u1'*X1mn; % produce the principal components projection
clearvars -except x11 y11 x21 y21 x31 y31 X1mn V1 D1 u1 s1 v1 Y1e Y1s W1s lambda1e

% Correlation Percentage 
figure()
plot(lambda1e(1:6)/sum(lambda1e),'*r','Linewidth',3) 
set(gca,'FontSize',24)
title('Normalized Diagonal Variance')
xlabel('Modes'), ylabel('Correlation')

% Modes
figure()
subplot(3,1,1)
plot(u1(:,1),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 1')

subplot(3,1,2)
plot(u1(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 2'), ylabel('Mode Value')

subplot(3,1,3)
plot(u1(:,3),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 3'), xlabel('Frames')

% Principal Components Projection 
figure()
subplot(3,2,1)
stem(Y1s(1:6,1),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 X')

subplot(3,2,3)
stem(Y1s(1:6,3),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 X'), ylabel('Mode Value')

subplot(3,2,5)
stem(Y1s(1:6,5),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 X'), xlabel('Modes')

subplot(3,2,2)
stem(Y1s(1:6,2),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 Y')

subplot(3,2,4)
stem(Y1s(1:6,4),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 Y')

subplot(3,2,6)
stem(Y1s(1:6,6),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 Y'), xlabel('Modes')
end 


%% Test 2: Noisy Data (Camera Shake)
function Test2 
% Load Camera 1 
load cam1_2.mat
v12 = vidFrames1_2(200:425,300:400,:,:); 
clear vidFrames1_2 
v12 = v12(:,:,:,96:end); 
v12size = size(v12,4); 
x12 = zeros(v12size,1); 
y12 = x12; 
for i = 1:v12size 
    v12g = rgb2gray(v12(:,:,:,i)); 
    focus = 20; 
    if i == 1 
        x12p = 31; 
        y12p = 130; 
    else 
        y12p = y12(i-1); 
        x12p = x12(i-1);
    end 
	v12g(floor(y12p+focus):end,:)=0; 
	v12g(1:floor(y12p-focus),:)=0; 
	v12g(:,floor(x12p+focus):end)=0; 
	v12g(:,1:floor(x12p-focus))=0; 
	[row,col] = find(v12g == max(max(v12g))); 
	x12(i)= median(col); 
	y12(i)= median(row); 
end 

% figure()
% subplot(1,3,1)
% plot(x12,'-r','Linewidth',2), hold on 
% plot(y12,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

% Load Camera 2
load cam2_2.mat 
v22 = vidFrames2_2(50:480,175:425,:,:); 
clear vidFrames2_2 
v22 = v22(:,:,:,43:end); 
v22size = size(v22,4); 
x22 = zeros(v22size,1); 
y22 = x22; 
for i = 1:v22size 
    v22g = rgb2gray(v22(:,:,:,i)); 
	focus = 25;
    if i == 1 
        x22p = 130; 
        y22p = 325; 
    else 
        y22p = y22(i-1); 
        x22p = x22(i-1); 
    end 
	v22g(floor(y22p+focus):end,:)=0; 
	v22g(1:floor(y22p-focus),:)=0; 
	v22g(:,floor(x22p+focus):end)=0; 
	v22g(:,1:floor(x22p-focus))=0; 
	[row,col] = find(v22g == max(max(v22g))); 
	x22(i)= median(col); 
	y22(i)= median(row); 
end

% subplot(1,3,2)
% plot(x22,'-r','Linewidth',2), hold on 
% plot(y22,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 2'), legend('X','Y'), xlabel('Frames')

% Load Camera 3 
load cam3_2.mat 
v32 = vidFrames3_2(150:350,250:525,:,:); 
v32 = v32(:,:,:,16:end); 
clear vidFrames3_2
v32size = size(v32,4); 
x32 = zeros(v32size,1); 
y32 = x32; 
for i = 1:v32size 
    v32g = rgb2gray(v32(:,:,:,i)); 
    focus =25; 
    if i == 1 
        x32p = 165; 
    	y32p = 98; 
    else 
    	y32p =y32(i-1); 
    	x32p =x32(i-1); 
    end 
	v32g(floor(y32p+focus):end,:)=0; 
	v32g(1:floor(y32p-focus),:)=0; 
	v32g(:,floor(x32p+focus):end)=0; 
	v32g(:,1:floor(x32p-focus))=0; 
	[row,col] = find(v32g == max(max(v32g))); 
	x32(i)= median(col); 
	y32(i)= median(row); 
end 

% subplot(1,3,3)
% plot(x32,'-r','Linewidth',2), hold on 
% plot(y32,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 3'), legend('X','Y')

% Reshape Data
minlen = min([size(x12,1),size(x22,1),size(x32,1)]);
X2 = [x12(1:minlen,:),y12(1:minlen,:),x22(1:minlen,:),... 
      y22(1:minlen,:),x32(1:minlen,:),y32(1:minlen,:)]; 
[m,n] = size(X2); 
X2mean = mean(X2,1); 
X2m = X2 - repmat(X2mean,m,1); 
X2mn = [X2m(:,1)/(sqrt(max(X2m(:,1))^2+max(X2m(:,2))^2)),... 
    	X2m(:,2)/(sqrt(max(X2m(:,1))^2+max(X2m(:,2))^2)),... 
    	X2m(:,3)/(sqrt(max(X2m(:,3))^2+max(X2m(:,4))^2)),... 
    	X2m(:,4)/(sqrt(max(X2m(:,3))^2+max(X2m(:,4))^2)),... 
    	X2m(:,5)/(sqrt(max(X2m(:,5))^2+max(X2m(:,6))^2)),... 
    	X2m(:,6)/(sqrt(max(X2m(:,5))^2+max(X2m(:,6))^2))]; 

figure()
subplot(1,3,1)
plot(X2mn(:,1),'-r','Linewidth',2), hold on 
plot(X2mn(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 1'), legend('X','Y'), ylabel('Normalized Pixel Location')

subplot(1,3,2)
plot(X2mn(:,3),'-r','Linewidth',2), hold on 
plot(X2mn(:,4),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 2'), legend('X','Y'), xlabel('Frames')

subplot(1,3,3)
plot(X2mn(:,5),'-r','Linewidth',2), hold on 
plot(X2mn(:,6),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 3'), legend('X','Y')

% Eignevalue & Eigenvector 
Cx2=(1/(n-1))*(X2mn*X2mn'); % covariance 
[V2,D2]=eig(Cx2); % eigenvectors(V)/eigenvalues(D)
lambda2e=diag(D2); % get eigenvalues 
[dummy,m_arrange2]=sort(-1*abs(lambda2e)); % sort in decreasing order 
lambda2e=lambda2e(m_arrange2); 

% SVD 
[u2,s2,v2] =svd(X2mn/sqrt(n-1),'econ'); % perform the SVD  
Y2s=u2'*X2mn; % produce the principal components projection
clearvars -except x12 y12 x22 y22 x32 y32 X2mn V2 D2 u2 s2 v2 Y2s Y2e W2s lambda2e 

% Correlation Percentage 
figure()
plot(lambda2e(1:6)/sum(lambda2e),'*r','Linewidth',3)
set(gca,'FontSize',24)
title('Normalized Diagonal Variance')
xlabel('Modes'), ylabel('Correlation')

% Modes
figure()
subplot(3,1,1)
plot(u2(:,1),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 1')

subplot(3,1,2)
plot(u2(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 2'), ylabel('Mode Value')

subplot(3,1,3)
plot(u2(:,3),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 3'), xlabel('Frames')

% Principal Components Projection 
figure()
subplot(3,2,1)
stem(Y2s(1:6,1),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 X')

subplot(3,2,3)
stem(Y2s(1:6,3),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 X'), ylabel('Mode Value')

subplot(3,2,5)
stem(Y2s(1:6,5),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 X'), xlabel('Modes')

subplot(3,2,2)
stem(Y2s(1:6,2),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 Y')

subplot(3,2,4)
stem(Y2s(1:6,4),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 Y')

subplot(3,2,6)
stem(Y2s(1:6,6),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 Y'), xlabel('Modes')
end


%% Test 3: Added Horizontal Component 
function Test3  
% Load Camera 1
load cam1_3.mat
v13 = vidFrames1_3(225:425 ,275:375 ,: ,:) ; 
clear vidFrames1_3
v13 = v13(:,:,:,16:150); 
v13size = size(v13,4); 
x13 = zeros(v13size,1); 
y13 = x13; 
for i = 1:v13size 
    v13g = rgb2gray(v13(:,:,:,i)); 
    focus = 15;
    if i == 1 
        x13p = 30;
        y13p = 130; 
    else 
        y13p = y13(i-1); 
        x13p = x13(i-1); 
    end 
	v13g(floor(y13p +focus):end,:)=0; 
	v13g(1:floor(y13p -focus),:)=0; 
	v13g(:,floor(x13p +focus):end)=0; 
	v13g(:,1:floor(x13p -focus))=0; 
	[row,col] = find(v13g == max(max(v13g))); 
	x13(i)= median(col); 
	y13(i)= median(row); 
end 

% figure()
% subplot(1,3,1)
% plot(x13,'-r','Linewidth',2), hold on 
% plot(y13,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

% Load Camera 2 
load cam2_3.mat
v23 = vidFrames2_3(50:480,175:425,:,:); 
clear vidFrames2_3 
v23 = v23(:,:,:,7:150); 
v23size = size(v23,4); 
x23 = zeros(v23size,1); 
y23 = x23;
for i = 1:v23size 
    v23g = rgb2gray(v23(:,:,:,i)); 
    focus = 25; 
    if i == 1 
        x23p = 65; 
        y23p = 242; 
    else 
        y23p = y23(i-1); 
        x23p = x23(i-1); 
    end 
    v23g(floor(y23p+focus):end,:)=0; 
    v23g(1:floor(y23p-focus),:)=0; 
    v23g(:,floor(x23p+focus):end)=0; 
    v23g(:,1:floor(x23p-focus))=0; 
    [row,col] = find(v23g == max(max(v23g))); 
    x23(i)= median(col); 
    y23(i)= median(row); 
end 

% subplot(1,3,2)
% plot(x23,'-r','Linewidth',2), hold on 
% plot(y23,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 2'), legend('X','Y'), xlabel('Frames')

% Load Camera 3 
load cam3_3.mat
v33 = vidFrames3_3(150:350,250:525,:,:); 
v33 = v33(:,:,:,16:150); 
clear vidFrames3_3 
v33size = size(v33,4); 
x33 = zeros(v33size,1); 
y33 = x33; 
for i = 1:v33size 
    v33g = rgb2gray(v33(:,:,:,i)); 
    focus = 25; 
    if i == 1 
        x33p = 100; 
        y33p = 80; 
    else 
        y33p = y33(i-1); 
        x33p = x33(i-1); 
    end 
    v33g(floor(y33p+focus):end,:)=0; 
    v33g(1:floor(y33p-focus),:)=0; 
    v33g(:,floor(x33p+focus):end)=0; 
    v33g(:,1:floor(x33p-focus))=0; 
    [row,col] = find(v33g == max(max(v33g))); 
    x33(i)= median(col); 
    y33(i)= median(row); 
end 

% subplot(1,3,3)
% plot(x33,'-r','Linewidth',2), hold on
% plot(y33,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 3'), legend('X','Y')

% Reshape Data 
minlen = min([size(x13,1),size(x23,1),size(x33,1)]);
X3 = [x13(1:minlen,:),y13(1:minlen,:),x23(1:minlen,:),... 
      y23(1:minlen,:),x33(1:minlen,:),y33(1:minlen,:)]; 
[m,n] = size(X3); 
X3mean = mean(X3,1); 
X3m = X3 - repmat(X3mean,m,1); 
X3mn = [X3m(:,1)/(sqrt(max(X3m(:,1))^2+max(X3m(:,2))^2)),... 
        X3m(:,2)/(sqrt(max(X3m(:,1))^2+max(X3m(:,2))^2)),... 
    	X3m(:,3)/(sqrt(max(X3m(:,3))^2+max(X3m(:,4))^2)),... 
    	X3m(:,4)/(sqrt(max(X3m(:,3))^2+max(X3m(:,4))^2)),... 
    	X3m(:,5)/(sqrt(max(X3m(:,5))^2+max(X3m(:,6))^2)),... 
    	X3m(:,6)/(sqrt(max(X3m(:,5))^2+max(X3m(:,6))^2))]; 

figure()
subplot(1,3,1)
plot(X3mn(:,1),'-r','Linewidth',2), hold on 
plot(X3mn(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

subplot(1,3,2)
plot(X3mn(:,3),'-r','Linewidth',2), hold on 
plot(X3mn(:,4),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 2'), legend('X','Y'), xlabel('Frames')

subplot(1,3,3)
plot(X3mn(:,5),'-r','Linewidth',2), hold on 
plot(X3mn(:,6),'-k','Linewidth',2)
set(gca,'FontSize',24)
legend('X','Y')
title('Camera 3')

% Eignevalue & Eigenvector 
Cx3=(1/(n-1))*(X3mn*X3mn'); % covariance 
[V3,D3]=eig(Cx3); % eigenvectors(V)/eigenvalues(D)
lambda3e=diag(D3); % get eigenvalues 
[dummy,m_arrange3]=sort(-1*abs(lambda3e)); % sort in decreasing order 
lambda3e=lambda3e(m_arrange3); 

% SVD 
[u3,s3,v3] =svd(X3mn/sqrt(n-1),'econ'); % perform the SVD 
Y3s=u3'*X3mn; % produce the principal components projection
clearvars -except x13 y13 x23 y23 x33 y33 X3mn V3 D3 u3 s3 v3 Y3s Y3e W3s lambda3e

% Correlation Percentage 
figure()
plot(lambda3e(1:6)/sum(lambda3e),'*r','Linewidth',3)
set(gca,'FontSize',24)
title('Normalized Diagonal Variance')
xlabel('Modes'), ylabel('Correlation')

% Modes
figure()
subplot(3,1,1)
plot(u3(:,1),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 1')

subplot(3,1,2)
plot(u3(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 2'), ylabel('Mode Value')

subplot(3,1,3)
plot(u3(:,3),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 3'), xlabel('Frames')

% Principal Components Projection 
figure()
subplot(3,2,1)
stem(Y3s(1:6,1),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 X')

subplot(3,2,3)
stem(Y3s(1:6,3),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 X'), ylabel('Mode Value')

subplot(3,2,5)
stem(Y3s(1:6,5),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 X'), xlabel('Modes')

subplot(3,2,2)
stem(Y3s(1:6,2),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 Y')

subplot(3,2,4)
stem(Y3s(1:6,4),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 Y')

subplot(3,2,6)
stem(Y3s(1:6,6),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 Y'), xlabel('Modes')
end


%% Test 4: Added Horizontal Displacement and Rotation
function Test4 
% Load Camera 1
load cam1_4.mat
v14 = vidFrames1_4(175:425,325:475,:,:); 
clear vidFrames1_4 
v14 = v14(:,:,:,34:end); 
v14size = size(v14,4); 
x14 = zeros(v14size,1); 
y14 = x14; 
for i = 1:v14size 
    v14g = rgb2gray(v14(:,:,:,i)); 
    focus = 15;
    if i == 1 
        x14p = 75; 
        y14p = 135; 
    else 
    	y14p = y14(i-1); 
    	x14p = x14(i-1); 
    end 
    v14g(floor(y14p +focus):end,:)=0; 
    v14g(1:floor(y14p -focus),:)=0; 
    v14g(:,floor(x14p +focus):end)=0; 
	v14g(:,1:floor(x14p -focus))=0; 
	[row,col] = find(v14g == max(max(v14g))); 
	x14(i)= median(col); 
	y14(i)= median(row); 
end

% figure()
% subplot(1,3,1)
% plot(x14,'-r','Linewidth',2), hold on 
% plot(y14,'-k','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

% Load Camera 2 
load cam2_4.mat 
v24 = vidFrames2_4(50:480,175:425,:,:); 
clear vidFrames2_4 
v24 = v24(:,:,:,44:end); 
v24size = size(v24,4); 
x24 = zeros(v24size,1); 
y24 = x24; 
for i = 1:v24size 
    v24g = rgb2gray(v24(:,:,:,i)); 
    focus = 20; 
    if i == 1 
        x24p = 85; 
        y24p = 240; 
    else 
        y24p = y24(i-1); 
        x24p = x24(i-1); 
    end 
    v24g(floor(y24p+focus):end,:)=0; 
    v24g(1:floor(y24p-focus),:)=0; 
    v24g(:,floor(x24p+focus):end)=0; 
    v24g(:,1:floor(x24p-focus))=0; 
    [row,col] = find(v24g == max(max(v24g)));
    x24(i)= median(col); 
    y24(i)= median(row); 
end 

% subplot(1,3,2)
% plot(x24,'-r','Linewidth',2), hold on
% plot(y24,'-b','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 2'), legend('X','Y'), xlabel('Frames')

% Load Camera 3 
load cam3_4.mat 
v34 = vidFrames3_4(150:350,250:525,:,:); 
v34 = v34(:,:,:,33:end); 
clear vidFrames3_4 
v34size = size(v34,4); 
x34 = zeros(v34size,1); 
y34 = x34; 
for i = 1:v34size 
    v34g = rgb2gray(v34(:,:,:,i)); 
    focus = 15; 
    if i == 1 
        x34p = 170; 
        y34p = 60; 
    else 
        y34p = y34(i-1); 
        x34p = x34(i-1); 
    end 
    v34g(floor(y34p+focus):end,:)=0; 
    v34g(1:floor(y34p-focus),:)=0; 
    v34g(:,floor(x34p+focus):end)=0; 
    v34g(:,1:floor(x34p-focus))=0;
    [row,col] = find(v34g == max(max(v34g)));
    x34(i)= median(col); 
    y34(i)= median(row); 
end 
 
% subplot(1,3,3)
% plot(x34,'-r','Linewidth',2), hold on 
% plot(y34,'-b','Linewidth',2)
% set(gca,'FontSize',24)
% title('Camera 3'), legend('X','Y')

% Reshape Data 
minlen = min([size(x14,1),size(x24,1),size(x34,1)]);
X4 = [x14(1:minlen,:),y14(1:minlen,:),x24(1:minlen,:),... 
      y24(1:minlen,:),x34(1:minlen,:),y34(1:minlen,:)]; 
[m,n] = size(X4); 
X4mean = mean(X4,1); 
X4m = X4 - repmat(X4mean,m,1); 
X4mn = [X4m(:,1)/(sqrt(max(X4m(:,1))^2+max(X4m(:,2))^2)),... 
        X4m(:,2)/(sqrt(max(X4m(:,1))^2+max(X4m(:,2))^2)),... 
        X4m(:,3)/(sqrt(max(X4m(:,3))^2+max(X4m(:,4))^2)),... 
        X4m(:,4)/(sqrt(max(X4m(:,3))^2+max(X4m(:,4))^2)),... 
        X4m(:,5)/(sqrt(max(X4m(:,5))^2+max(X4m(:,6))^2)),... 
        X4m(:,6)/(sqrt(max(X4m(:,5))^2+max(X4m(:,6))^2))]; 
 
figure()
subplot(1,3,1)
plot(X4mn(:,1),'-r','Linewidth',2), hold on 
plot(X4mn(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 1'), legend('X','Y'), ylabel('Pixel Location')

subplot(1,3,2)
plot(X4mn(:,3),'-r','Linewidth',2), hold on 
plot(X4mn(:,4),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 2'), legend('X','Y'), xlabel('Frames')

subplot(1,3,3)
plot(X4mn(:,5),'-r','Linewidth',2), hold on 
plot(X4mn(:,6),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Camera 3'), legend('X','Y')

% Eignevalue & Eigenvector 
Cx4=(1/(n-1))*(X4mn*X4mn'); % covariance 
[V4,D4]=eig(Cx4); % eigenvectors(V)/eigenvalues(D)
lambda4e=diag(D4); % get eigenvalues 
[dummy,m_arrange4]=sort(-1*abs(lambda4e)); % sort in decreasing order 
lambda4e=lambda4e(m_arrange4); 

% SVD 
[u4,s4,v4] =svd(X4mn/sqrt(n-1),'econ'); % perform the SVD 
Y4s=u4'*X4mn; % produce the principal components projection
clearvars -except x14 y14 x24 y24 x34 y34 X4mn V4 D4 u4 s4 v4 Y4s Y4e W4s lambda4e

% Correlation Percentage 
figure()
plot(lambda4e(1:6)/sum(lambda4e),'*r','Linewidth',3)
set(gca,'FontSize',24)
title('Normalized Diagonal Variance')
xlabel('Modes'), ylabel('Correlation')

% Modes
figure()
subplot(3,1,1)
plot(u4(:,1),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 1')

subplot(3,1,2)
plot(u4(:,2),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 2'), ylabel('Mode Value')

subplot(3,1,3)
plot(u4(:,3),'-k','Linewidth',2)
set(gca,'FontSize',24)
title('Mode 3'), xlabel('Frames')

% Principal Components Projection 
figure()
subplot(3,2,1)
stem(Y4s(1:6,1),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 X')

subplot(3,2,3)
stem(Y4s(1:6,3),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 X'), ylabel('Mode Value')

subplot(3,2,5)
stem(Y4s(1:6,5),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 X'), xlabel('Modes')

subplot(3,2,2)
stem(Y4s(1:6,2),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 1 Y')

subplot(3,2,4)
stem(Y4s(1:6,4),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 2 Y')

subplot(3,2,6)
stem(Y4s(1:6,6),'ok','Linewidth',3)
set(gca,'FontSize',24)
title('Cam 3 Y'), xlabel('Modes')
end
