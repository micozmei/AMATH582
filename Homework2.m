% Homework 2
clear all; close all; clc

%% Problem 1
% Load Handel's Messiah
load handel
v = y'/2; 
figure()
plot((1:length(v))/Fs,v,'k'); 
xlabel('Time(s)'); ylabel('S(t)'); 
title('Signal of Interest'); 

% Resize and Discretize Domain 
vr = v(1:end-1); 
t =(1:length(vr))/Fs; 
L = t(end); %Domain
n = length(vr); 
k = (2*pi/L)*[0:n/2-1, -n/2:-1]; %Wavenumbers
ks = fftshift(k); 
vt = fft(vr); 

% Plot Signal of Interest in Time and Frequency Domain
figure()
subplot(2,1,1) 
plot(t,vr,'k','Linewidth',[2])
xlabel('Time(s)'); ylabel('S(t)'); 
title('Signal of Interest')
subplot(2,1,2) 
plot(fftshift(k), abs(fftshift(vt)),'k','Linewidth',[2])
xlabel('Frequency(\omega)'); ylabel('FFT(S(t))'); 

% Gaussian Filter Width Study 
figure()
a = [1,100]; %Width 
for i = 1:length(a) 
    g = exp(-a(i)*(t-2).^2); 
    subplot(length(a),1,i) 
    plot(t,vr,'k'), hold on 
    plot(t,g,'r','Linewidth',[2])
    set(gca,'Fontsize',[14])
    xlabel('Time(s)'); ylabel('S(t), g(t)');
    axis([0 8.9 -1 1])
    legend(['Signal'],['a=',num2str(a(i))])
    title('Gaussian Filter e^{-a(t)^2} at t=2')
end 

figure()
g = exp(-(t-4.5).^2); 
vg = g.*vr; 
vgt = fft(vg); 

subplot(3,1,1), plot(t,vr,'k'), hold on 
plot(t,g,'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
title('Signal of Interest with Gaussian Filter (width a=1 at t=4.5s)')
axis([0, 9, -1, 1])

subplot(3,1,2), plot(t,vg,'k')
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t)*g(t)');
axis([0, 9, -0.4, 0.4])

subplot(3,1,3), plot(ks,abs(fftshift(vgt))/max(abs(vgt)),'k')
set(gca,'Fontsize',[14])
xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))');
axis([-2.5e4, 2.5e4, 0, 1])

% Gaussian Filter 
for i = 1:length(a)
    bn = [8,32,128]; %Translational Discretization Parameter
    for j = 1: length(bn) 
        b=linspace(0,t(end),bn(j)); 
        vgt_spect =[];
        figure()
        for k =1:length(b) 
            gt=exp(-a(i)*(t-b(k)).^2); %Gabor 
            vg=gt.*vr;
            vgt=fft(vg); 
            vgt_spect=[vgt_spect; abs(fftshift(vgt))]; 
            
            subplot(3,1,1), plot(t, vr,'k',t, gt,'r')
            title(['Signal of Interest with Filter Width a=',num2str(a(i)),' and b=',num2str(bn(j))])
            xlabel('Time(s)'); ylabel('S(t), g(t)'); 
            
            subplot(3,1,2), plot(t,vg,'k')
            xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
            
            subplot(3,1,3), plot(ks, abs(fftshift(vgt))/max(abs(vgt)),'k')
            xlabel('Frequency(\omega)'); ylabel('FFT(Sg)'); 
            drawnow, pause(0.01) 
        end 
        figure()
        pcolor(b,ks, vgt_spect.'), shading flat 
        set(gca,'Fontsize',[14])
        colormap(hot) 
        xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
        title(['Spectrogram Analysis of Signal of Interest (a=', num2str(a(i)),' and b=', num2str(bn(j)),')'])
    end 
end 
    

figure()
% Mexican Hat Filter
[mhm, tval] = mexihat(t(1)-4.5,t(end)-4.5,n); 
% Gaussian Filter
g = exp(-(t-4.5).^2);
% Step Filter
step = zeros(1, length(vr)); 
for i = 1: length(vr)
    if abs(t(i)-4.5)<0.5 
        step(i)=1; 
    else 
        step(i)=0; 
    end 
end 
% Harr Filter
harr = zeros(1, length(vr)); 
for i = 1: length(vr) 
    if t(i)>4.5 && t(i)<4.5+0.5 
        harr(i)=-1; 
    elseif t(i)<4.5 && t(i)>4.5-0.5 
        harr(i)=1; 
    else 
        harr(i)=0; 
    end 
end
% Combined Filters
convolution1 = conv(harr, step,'same'); 
convolution2 = conv(harr,g,'same'); 

subplot(3,2,1) 
plot(t, vr,'k'), hold on 
plot(t,g,'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
%legend('Signal','Gaussian')
%title('Filters Study')

subplot(3,2,2) 
plot(t, vr,'k'), hold on 
plot(t,mhm,'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
%legend('Signal','Mexican Hat')

subplot(3,2,3) 
plot(t, vr,'k'), hold on
plot(t, step,'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
%legend('Signal','Square')

subplot(3,2,4) 
plot(t, vr,'k'), hold on 
plot(t, harr,'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
%legend('Signal','Harr')

subplot(3,2,5) 
plot(t, vr,'k'), hold on 
plot(t, convolution1/(max(convolution1)),'r','Linewidth',[2])
set(gca,'Fontsize',[14])
xlabel('Time(s)'); ylabel('S(t), g(t)');
%conv1 = legend('Signal','Harr $\ast$ Square'); 
%set(conv1,'Interpreter','latex')

subplot(3,2,6) 
plot(t, vr,'k'), hold on 
plot(t, convolution2/(max(convolution2)),'r','Linewidth',[2])
set(gca,'Fontsize',[14]) 
xlabel('Time(s)'); ylabel('S(t), g(t)');
%conv2 = legend('Signal','Harr $\ast$ Gaussian'); 
%set(conv2,'Interpreter','latex')
axis([0, 9, -1, 1])

% Applying the Different Filters
% Gaussian 
b=linspace(0,t(end),128); %Translational Discretization Parameter
vgt_spect=[]; 

figure()
for i=1:length(b) 
    gt=exp(-1*(t-b(i)).^2); %Gabor 
    vg=gt.*vr;
    vgt=fft(vg); 
    vgt_spect=[vgt_spect; abs(fftshift(vgt))]; 
    
    subplot(3,1,1), plot(t, vr,'k',t, gt,'r')
    title(['Signal of Interest with Gaussain Filter (a=1 and b=128)'])
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    
    subplot(3,1,2), plot(t,vg,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(ks, abs( fftshift(vgt))/max(abs(vgt)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow, pause(0.01) 
end

figure()
pcolor(b, ks, vgt_spect.'), shading flat 
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title(['Spectrogram Analysis of Signal with Gaussian Filter (a=1 and b=128)'])

% Mexican Hat 
vmt_spect =[]; 
figure()
for i=1:length(b) 
    [mhm, tval] = mexihat(t(1)-b(i),t(end)-b(i),n); 
    vm=mhm.*vr; 
    vmt=fft(vm);
    vmt_spect=[vmt_spect; abs(fftshift(vmt))]; 
    
    subplot(3,1,1), plot(t,vr,'k',t,mhm,'r')
    title('Signal of Interest with Mexican Hat Filter (a=1 and b=128)')
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    
    subplot(3,1,2), plot(t,vm,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(ks,abs(fftshift(vmt))/max(abs(vmt)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow
    pause(0.01) 
end

figure()
pcolor(b, ks, vmt_spect.'), shading flat 
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title('Spectrogram Analysis of Signal with Mexican Hat Filter (a=1 and b=128)'); 

% Square Filter 
vst_spect =[]; 
step = zeros(1, length(vr)); 
figure()
for i=1:length(b) 
    for j =1:length(vr) 
        if abs(t(j)-b(i))<0.5 
            step(j)=1; 
        else
            step(j)=0;
        end
    end
    vs=step.*vr; 
    vst=fft(vs); 
    vst_spect=[vst_spect; abs(fftshift(vst))]; 
    
    subplot(3,1,1), plot(t, vr,'k',t, step,'r')
    title('Signal of Interest with Square Filter (a=1 and b=128)') 
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    axis([0, 9, -1, 1]) 
    
    subplot(3,1,2), plot(t,vs,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(ks, abs(fftshift(vst))/max(abs(vst)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow, pause(0.01) 
end

figure()
pcolor(b, ks, vst_spect.'), shading flat 
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title('Spectrogram Analysis of Signal with Square Filter (a=1 and b=128)'); 

% Harr Filter 
vht_spect =[]; 
harr = zeros(1,length(vr)); 

figure()
for i=1:length(b) 
    for j = 1:length(vr) 
        if t(j)>b(i) && t(j)<b(i)+0.5 
            harr(j)=-1; 
        elseif t(j)<b(i) && t(j)>b(i)-0.5 
            harr(j)=1; 
        else 
            harr(j)=0; 
        end 
    end 
    vh=harr.*vr; 
    vht=fft(vh); 
    vht_spect=[vht_spect; abs(fftshift(vht))]; 
    
    subplot(3,1,1), plot(t, vr,'k',t, harr,'r')
    title('Signal of Interest with Harr Filter (a=1 and b=128)')
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    axis([0 9 -1 1])
    
    subplot(3,1,2), plot(t,vh,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(ks, abs( fftshift(vht))/max(abs(vht)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow, pause(0.01) 
end

figure()
pcolor(b, ks, vht_spect.'), shading flat
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title('Spectrogram Analysis of Signal with Harr Filter (a=1 and b=128)'); 

% Convolution 1 (Harr*Square)
vcon1t_spect =[]; 
figure()
for i=1:length(b) 
    % Harr
    for j = 1: length(vr) 
        if t(j)>b(i) && t(j)<b(i)+0.5 
            harr(j)=-1; 
        elseif t(j)<b(i) && t(j)>b(i)-0.5 
            harr(j)=1; 
        else 
            harr(j)=0; 
        end 
    end
    % Step
    for j =1:length(vr)
        if abs(t(j)-b(i))<0.5 
            step(j)=1; 
        else 
            step(j)=0; 
        end 
    end 
    convolution1 = conv(harr, step,'same'); 
    convolution1 = convolution1/max(convolution1); 
    vcon1 = convolution1.*vr; 
    vcon1t=fft(vcon1); 
    vcon1t_spect=[vcon1t_spect; abs(fftshift(vcon1t))]; 
    
    subplot(3,1,1), plot(t, vr,'k',t, convolution1,'r')
    title('Signal of Interest with Convolution of Harr and Square Filter (a=1 and b=128)')
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    axis([0, 9, -1, 1])
    
    subplot(3,1,2), plot(t, vcon1,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)');
    
    subplot(3,1,3), plot(ks, abs( fftshift(vcon1t))/max(abs( vcon1t)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow, pause(0.01) 
end

figure()
pcolor(b, ks, vcon1t_spect.'), shading flat 
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
%title('Spectrogram Analysis of Signal with Convolution of Harr and Square Filter (a=1 and b=128)'); 

% Convolution 2 (Harr*Gaussian)
vc2t_spect =[]; 
figure()
for i=1:length(b) 
    % Harr
    for j = 1: length(vr) 
        if t(j)>b(i) && t(j)<b(i)+0.5 
            harr(j)=-1; 
        elseif t(j)<b(i) && t(j)>b(i)-0.5 
            harr(j)=1; 
        else 
            harr(j)=0; 
        end 
    end 
    gt=exp(-0.5*(t-b(i)).^2); 
    convolution2 = conv(harr,g,'same'); 
    convolution2 = convolution2/max(convolution2); 
    vcon2 = convolution2.*vr; 
    vcon2t=fft(vcon2); 
    vc2t_spect=[vc2t_spect; abs( fftshift(vcon2t))]; 
    
    subplot(3,1,1), plot(t, vr,'k',t, convolution2,'r')
    title('Signal of Interest with Convolution of Harr and Gaussian Filter (a=1 and b=128)')
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    axis([0, 9, -1, 1])
    
    subplot(3,1,2), plot(t, vcon2,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(ks, abs( fftshift(vcon2t))/max(abs( vcon2t)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t)*g(t))'); 
    drawnow, pause(0.01) 
end

figure()
pcolor(b, ks, vc2t_spect.'), shading flat 
set(gca,'Fontsize',[14])
colormap(hot) 
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
%title('Spectrogram Analysis of Signal with Convolution of Harr and Gaussian Filter (a=1 and b=128)'); 


%% Problem 2 
% Audio 1 
tr_piano = 16; %Record time (seconds) 
y1 = audioread('music1.wav')'; 
Fs1 = length(y1)/tr_piano; 
t1 = (1: length(y1))/Fs1; 

figure() 
plot(t1,y1,'k'); 
xlabel('Time(s)'); ylabel('S(t)'); 
title('Mary Had a Little Lamb (Piano)'); drawnow 
p81 = audioplayer(y1, Fs1); 
%playblocking(p81); 

L1 = t1(end); %Domain
n1 =length(y1); 
k1 =(2*pi/L1)*[0:n1/2-1, -n1/2:-1]; %Wavenumbers
k1s = fftshift(k1); 
y1t = fft(y1); 

figure()
subplot(2,1,1) 
plot(t1,y1,'k','Linewidth',[2])
title('Mary Had a Little Lamb (Piano)')
xlabel('Time(s)'); ylabel('S(t)'); 

subplot(2,1,2) 
plot(fftshift(k1),abs(fftshift(y1t)),'k','Linewidth',[2])
xlabel('Frequency(\omega)'); ylabel('FFT(S(t))'); 

b1 = linspace(0, t1(end),128); 
y1gt_spect = []; 

figure()
a1 = 100; %Width
for i = 1:length(b1) 
    gt1 = exp(-a1*(t1-b1(i)).^2); %Gabor 
    y1g = gt1.*y1; 
    y1gt = fft(y1g); 
    y1gt_spect = [y1gt_spect; abs(fftshift(y1gt))]; 
    
    subplot(3,1,1), plot(t1,y1,'k',t1, gt1,'r')
    axis([0, L1, -1, 1])
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    title('Mary Had a Little Lamb with Filter (Piano)')
    
    subplot(3,1,2), plot(t1, y1g,'k')
    axis([0, L1, -1, 1])
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    
    subplot(3,1,3), plot(k1s, abs(fftshift(y1gt))/max(abs(y1gt)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t))'); 
    drawnow, pause(0.01) 
end 

% Audio 1 Plotting 
figure()
pcolor(b1, k1s, y1gt_spect.'/(max(max( y1gt_spect.')))), shading interp 
set(gca,'Fontsize',[14])
colormap(hot)
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title('Spectrogram Analysis of Mary Had a Little Lamb (Piano)')
axis([0, L1, 250*(2*pi), 350*(2*pi)])

figure()
pcolor(b1, k1s /(2*pi), y1gt_spect.'/(max(max(y1gt_spect.')))), shading interp 
set(gca,'Fontsize',[14])
colormap(hot)
xlabel('Time(s)'); ylabel('Frequency(Hz)'); 
title('Spectrogram Analysis of Mary Had a Little Lamb (Piano)')
axis([0, L1, 250, 350])

% Audio 2 
tr_rec = 14; %Record time (seconds)
y2 = audioread('music2.wav')'; 
Fs2 = length(y2)/tr_rec; 
t2 = (1:length(y2))/Fs2; 

figure() 
plot(t2,y2,'k'); 
xlabel('Time(s)'); ylabel('S(t)'); 
title('Mary Had a Little Lamb (Recorder)'); 
p82 = audioplayer(y2, Fs2);

L2 = t2(end); %Domain
n2 = length(y2); 
k2 =(2*pi/L2)*[0: n2/2-1 -n2/2:-1]; %Wavenumbers
k2s = fftshift(k2); 
y2t = fft(y2); 

figure()
subplot(2,1,1) 
plot(t2,y2,'k','Linewidth',[2])
xlabel('Time(s)'); ylabel('S(t)'); 
title('Mary Had a Little Lamb (Recorder)')

subplot(2,1,2) 
plot(fftshift(k2),abs(fftshift(y2t)),'k','Linewidth',[2])
xlabel('Time(s)'); ylabel('FFT(S(t))'); 
title('Mary Had a Little Lamb (Recorder)')

b2 = linspace(0, t1(end),128); 
y2gt_spect = []; 

figure()
a2 = 100; %Width
for i = 1:length(b2) 
    gt2 = exp(-a2*(t2-b2(i)).^2); %Gabor 
    y2g = gt2.*y2; 
    y2gt = fft(y2g); 
    y2gt_spect = [y2gt_spect; abs(fftshift(y2gt))]; 
    
    subplot(3,1,1), plot(t2,y2,'k',t2, gt2,'r')
    xlabel('Time(s)'); ylabel('S(t), g(t)'); 
    title('Mary Had a Little Lamb(Recorder)')
    axis([0, L2, -1, 1]) 
    
    subplot(3,1,2), plot(t2, y2g,'k')
    xlabel('Time(s)'); ylabel('S(t)*g(t)'); 
    axis([0, L2, -1, 1])
    
    subplot(3,1,3), plot(k2s, abs(fftshift(y2gt))/max(abs(y2gt)),'k')
    xlabel('Frequency(\omega)'); ylabel('FFT(S(t))'); 
    drawnow, pause(0.01) 
end

% Audio 2 Plotting
figure()
pcolor(b2, k2s, y2gt_spect.'/(max(max(y2gt_spect.')))), shading interp 
set(gca,'Fontsize',[14])
colormap(hot)
xlabel('Time(s)'); ylabel('Frequency(\omega)'); 
title('Spectrogram Analysis of Mary Had a Little Lamb (Recorder)')
axis([0, L2, 800*(2*pi), 1100*(2*pi)])

figure()
pcolor(b2, k2s/(2*pi), y2gt_spect.'/(max(max(y2gt_spect.')))), shading interp 
set(gca,'Fontsize',[14])
colormap(hot)
xlabel('Time(s)'); ylabel('Frequency(Hz)'); 
title('Spectrogram Analysis of Mary Had a Little Lamb (Recorder)')
axis([0, L2, 800, 1100])
