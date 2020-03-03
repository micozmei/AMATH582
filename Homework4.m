% Homework #4
clear all; close all; clc;

%Question1_Part1 
%Question1_Part2

%Question2_Part1
%Question2_Part2 
Question2_Part3 
%Question2_Stats 

%% Question 1 Part 1 
function Question1_Part1 
    % Cropped 
    yaleBXX = dir('C:\Users\skewl\Desktop\CroppedYale');
    A_uint8 = [];
    for i = 3:length(yaleBXX)
        yaleBXX_X = dir(strcat('C:\Users\skewl\Desktop\CroppedYale','\',yaleBXX(i).name)); 
        if~isempty(yaleBXX_X)
            for j = 3:length(yaleBXX_X)
                A8ij = imread(strcat('C:\Users\skewl\Desktop\CroppedYale','\',yaleBXX(i).name,'\',yaleBXX_X(j).name)); 
                A8i = A8ij(:);
                A_uint8 = [A_uint8,A8i];
            end 
        end 
    end 
    A = double(A_uint8); 
    clearvars -except A 

    % SVD
    [u1,s1,v1] = svd(A,'econ'); 
    diags1 = diag(s1)/sum(diag(s1)); 

    % Plots     
    % Log-scale Singular Value 
    figure()
    semilogy(diags1,'k*','Linewidth',2)
    title('Singular Values','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('Value','fontsize',30)
    grid on 

    % Modes 
    figure()
    for i=1:8 
        subplot(2,4, i); 
        imshow(reshape(u1(:,i),192,168),[])
        title(['Mode ',num2str(i)],'fontsize',20); 
    end

    % 85% 
    figure()
    plot(1:length(s1),0.85*ones(length(s1)),'r-','Linewidth',2)
    hold on 
    percent1=zeros(length(s1),1); 
    for i = 1:length(s1)
        percent1(i)=sum(diags1(1:i)); 
    end 
    plot(1:length(s1),percent1,'k*')
    hold on 
    title('Modes vs % Reconstruction','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('% Reconstruction','fontsize',30)
    grid on 

    % Reconstruction 
    figure()
    image_nubmber1=1;
    subplot(2,4,1);
    imshow(reshape(A(:,image_nubmber1),192,168),[])
    subplot(2,4,1);title('Original','fontsize',20);
    percent=0.1:0.15:1; % 15% Increments
    coln=[]; 
    for i=1: length(percent)
        [row,col]=find(percent1>percent(i)); 
        coln=[coln,col(1)];
    end 
    for i=1:length(coln)
        USVi1=u1(:,1: coln(i))*s1(1: coln(i),1: coln(i))*v1(:,1:coln(i))'; 
        USVij1=reshape(USVi1(:,image_nubmber1),192,168); 
        subplot(2,4,i+1)
        imshow(USVij1,[])
        subplot(2,4,i+1);title([num2str(percent(i)*100),'%'],'fontsize',20); 
    end 
end


%% Question 1 Part 2
function Question1_Part2 
    % Uncropped 
    yaleface = dir('C:\Users\skewl\Desktop\yalefaces'); 
    B_uint8 = []; 
    for i = 3:length(yaleface) 
        B8ij = imread(strcat('C:\Users\skewl\Desktop\yalefaces','\',yaleface(i).name)); 
        B8i = B8ij(:); 
        B_uint8=[B_uint8,B8i]; 
    end
    B = double(B_uint8); 
    clearvars -except B

    % SVD 
    [u2,s2,v2] = svd(B,'econ'); 
    diags2 = diag(s2)/sum(diag(s2)); 

    % Plots 
    % Log-scale Singular Value 
    figure()
    semilogy(diags2,'k*','Linewidth',2)
    title('Singular Values','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('Value','fontsize',30)
    grid on 

    % Modes 
    figure()
    for i=1:8
        subplot(2,4,i); 
        imshow(reshape(u2(:,i),243,320),[])
        title(['Mode ', num2str(i)],'fontsize',20); 
    end

    % 85% 
    figure()
    plot(1: length(s2),0.85*ones(length(s2)),'r-','Linewidth',2)
    hold on 
    percent2=zeros(length(s2),1); 
    for i=1:length(s2)
        percent2(i)=sum(diags2(1:i)); 
    end 
    
    plot(1:length(s2),percent2,'k*')
    hold on
    title('Modes vs % Reconstruction','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('% Reconstruction','fontsize',30)
    grid on

    % Reconstruction 
    figure()
    image_nubmber2=1; 
    subplot(2,4,1)
    imshow(reshape(B(:,image_nubmber2),243,320),[])
    subplot(2,4,1); title('Original','fontsize',20); 
    percent = 0.25:0.1:0.85; % 10% Increments
    
    coln =[]; 
    for i = 1:length(percent)
        [row,col] = find(percent2>percent(i)); 
        coln = [coln,col(1)];
    end 
    
    for i=1:length(coln)
        USVi2=u2(:,1:coln(i))*s2(1:coln(i),1:coln(i))*v2(:,1:coln(i))'; 
        USVij2=reshape(USVi2(:,image_nubmber2),243,320); 
        subplot(2,4,i+1)
        imshow(USVij2,[])
        subplot(2,4,i+1);title([num2str(percent(i)*100),'%'],'fontsize',20); 
    end 
end


%% Test 1 Band Classification 
function Question2_Part1 
    n = 5; % sample size

    % Electronic: Deadmau5 
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Deadmau5\1_Faxing_Berlin.mp3'); 
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Deadmau5\2_Sofi_Needs_A_Ladder.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Deadmau5\3_Right_This_Second.mp3');
    [spec11]= mus2spect(y11, fs11, n); 
    [spec12]= mus2spect(y12, fs12, n); 
    [spec13]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Classical: Mozart
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Mozart\1_Eine_Kleine_Nachtmusik.mp3'); 
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Mozart\2_Sonata_16.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Mozart\3_Rondo_Alla_Turca.mp3');
    [spec14]= mus2spect(y11, fs11, n); 
    [spec15]= mus2spect(y12, fs12, n);
    [spec16]= mus2spect(y13, fs13, n);
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Black Metal: Bathory
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Bathory\1_A_Fine_Day_to_Die.mp3');
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Bathory\2_For_All_Those_Who_Died.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Bathory\3_Necromancy.mp3'); 
    [spec17]= mus2spect(y11, fs11, n); 
    [spec18]= mus2spect(y12, fs12, n); 
    [spec19]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % SVD 
    D5_size = size([spec11, spec12, spec13]); 
    Moz_size = size([spec14, spec15, spec16]); 
    B_size = size([spec17, spec18, spec19]);
    spec1 = [spec11, spec12, spec13, spec14, spec15, spec16, spec17, spec18, spec19]; 
    clearvars -except spec1 D5_size Moz_size B_size 
    
    [u1,s1,v1] = svd(spec1,'econ');
    diags1 = diag(s1)/sum(diag(s1));

    figure()
    plot(diags1,'k*','Linewidth',2)
    title('Test 1 Singular Values','fontsize',30);
    xlabel('Singular Value','fontsize',30), ylabel('Fractional Value','fontsize',30)
    grid on 

    figure()
    plot(1:length(s1), 0.85*ones(length(s1)),'r-','Linewidth',2)
    hold on 

    percent1 = zeros(length(s1),1); 
    for i = 1:length(s1)
        percent1(i) = sum(diags1(1:i)); 
    end

    plot(1:length(s1), percent1,'k*'), hold on 
    title('Test 1 Modes vs % Reconstruction','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('% Reconstruction','fontsize',30)
    grid on 

    percent = 0.1:0.1:0.9; 
    coln = []; 
    for i = 1:length(percent)
        [row,col] = find(percent1>percent(i)); 
        coln = [coln,col(1)];
    end
    
    % Classification 
    Positive1 = []; 
    for i = 1:1000
        nt = 0.5; % for train data 
        nmin = coln(2); % reconstruction lower bound 
        nmax = coln(8); % upper bound 
        n_Deadmau5 = round(D5_size(2)*nt); 
        n_Mozart = round(Moz_size(2)*nt); 
        n_Bathory = round(D5_size(2)*nt);
        q_Deadmau5 = randperm(D5_size(2)); 
        q_Mozart = randperm(Moz_size(2)); 
        q_Bathory = randperm(B_size(2)); 
        x_Deadmau5 = v1(1:D5_size(2),:);
        x_Mozart = v1(D5_size(2)+1:D5_size(2)+Moz_size(2),:);
        x_Bathory = v1(D5_size(2)+Moz_size(2)+1:D5_size(2)+Moz_size(2)+B_size(2),:);
        xtrain1 = [x_Deadmau5(q_Deadmau5(1:n_Deadmau5),:); 
        x_Mozart(q_Mozart(1:n_Mozart),:); 
        x_Bathory(q_Bathory(1:n_Bathory),:)]; 
        xtest1 = [x_Deadmau5(q_Deadmau5(n_Deadmau5+1:end),:); 
        x_Mozart(q_Mozart(n_Mozart+1:end),:); 
        x_Bathory(q_Bathory(n_Bathory+1:end),:)]; 
        ctrain1 = [ones(n_Deadmau5,1); 2*ones(n_Mozart,1); 3*ones(n_Bathory,1)]; 
        xtrain1 = real(xtrain1); % Train data
        xtest1 = real(xtest1); % Test data
        ctrain1 = real(ctrain1); % Label train data
        nb1 = fitcnb(xtrain1, ctrain1); % Naive Bayes
        pre1 = nb1.predict(xtest1); % Prediction
        qD5s = size(q_Deadmau5(n_Deadmau5+1:end)); 
        qMozs = size(q_Mozart(n_Mozart+1:end)); 
        qBs = size(q_Bathory(n_Bathory+1:end)); 
        positive1 = [sum(pre1(1:qD5s(2))==1); sum(pre1(qD5s(2)+1:qD5s(2)+qMozs(2))==2); sum(pre1(qD5s(2)+qMozs(2)+1:end)==3)]; 
        Positive1 = [Positive1,positive1]; 
    end 
    clearvars -except spec1 D5_size Moz_size B_size Positive1 qD5s qMozs qBs pre1 
    
    accuracy1 = [mean(Positive1(1,:))/qD5s(2), mean(Positive1(2,:))/qMozs(2), mean(Positive1(3,:))/qBs(2)]; 
    accuracy1

    figure()
    bar(accuracy1,'k')
    title('Test 1 Classification Accuracy','fontsize',30)
end


%% Test 2 Same Genre Classification 
function Question2_Part2 
    n = 5; % sample size 
    
    % Black Metal: Bathory
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Bathory\1_A_Fine_Day_to_Die.mp3');
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Bathory\2_For_All_Those_Who_Died.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Bathory\3_Necromancy.mp3'); 
    [spec11]= mus2spect(y11, fs11, n); 
    [spec12]= mus2spect(y12, fs12, n);
    [spec13]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Black Metal: Darkthrone
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Darkthrone\1_Transilvanian_Hunger.mp3');
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Darkthrone\2_The_Pagan_Winter.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Darkthrone\3_Where_Cold_Winds_Blow.mp3'); 
    [spec14]= mus2spect(y11, fs11, n); 
    [spec15]= mus2spect(y12, fs12, n); 
    [spec16]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Black Metal: Mayhem
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Mayhem\1_Freezing_Moon.mp3');
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Mayhem\2_Pagan_Fears.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Mayhem\3_Life_Eternal.mp3');
    [spec17]= mus2spect(y11, fs11, n); 
    [spec18]= mus2spect(y12, fs12, n); 
    [spec19]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 
   
    % SVD 
    B_size = size([spec11, spec12, spec13]); 
    D_size = size([spec14, spec15, spec16]); 
    M_size = size([spec17, spec18, spec19]); 
    spec2 = [spec11, spec12, spec13, spec14, spec15, spec16, spec17, spec18, spec19]; 
    clearvars -except spec2 B_size D_size M_size 
    
    [u2,s2,v2] = svd(spec2,'econ'); 
    diags2 = diag(s2)/sum(diag(s2)); 

    figure()
    plot(diags2,'k*','Linewidth',2)
    title('Test 2 Singular Values','fontsize',30);
    xlabel('Singular Value','fontsize',30), ylabel('Fractional Value','fontsize',30)
    grid on 

    figure()
    plot(1: length(s2), 0.85*ones(length(s2)),'r-','Linewidth',2)
    hold on 

    percent2 = zeros(length(s2),1); 
    for i = 1: length(s2)
        percent2(i) = sum(diags2(1:i)); 
    end

    plot(1:length(s2), percent2,'k*'), hold on
    title('Test 2 Modes vs % Reconstruction','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('% Reconstruction','fontsize',30)
    grid on

    percent = 0.1:0.1:0.9; 
    coln = []; 
    for i = 1:length(percent)
        [row,col] = find(percent2>percent(i)); 
        coln = [coln,col(1)];
    end

    % Classification
    Positive2 = []; 
    for i = 1:1000 
        nt = 0.5; % for train data 
        nmin = coln(2); % reconstruction lower bound 
        nmax = coln(8); % upper bound 
        n_Bathory = round(B_size(2)*nt); 
        n_Darkthrone = round(D_size(2)*nt); 
        n_Mayhem = round(M_size(2)*nt); 
        q_Bathory = randperm(B_size(2)); 
        q_Darkthrone = randperm(D_size(2)); 
        q_Mayhem = randperm(M_size(2)); 
        x_Bathory = v2(1: B_size(2), :);
        x_Darkthrone = v2(B_size(2)+1:B_size(2)+D_size(2),:);
        x_Mayhem = v2(B_size(2)+D_size(2)+1:B_size(2)+D_size(2)+M_size(2), :);
        xtrain2 =[x_Bathory(q_Bathory(1:n_Bathory),:); x_Darkthrone(q_Darkthrone(1:n_Darkthrone),:); x_Mayhem(q_Mayhem(1:n_Mayhem),:)]; 
        xtest2 =[x_Bathory(q_Bathory(n_Bathory+1:end),:); x_Darkthrone(q_Darkthrone(n_Darkthrone+1:end),:); x_Mayhem(q_Mayhem(n_Mayhem+1:end),:)]; 
        ctrain2 =[ones(n_Bathory,1); 2*ones(n_Darkthrone,1); 3*ones(n_Mayhem,1)]; 
        xtrain2 = real(xtrain2); % Train data
        xtest2 = real(xtest2); % Test data
        ctrain2 = real(ctrain2); % Label train data
        nb2 = fitcnb(xtrain2, ctrain2); % Naive Bayes
        pre2 = nb2.predict(xtest2); % Prediction
        qBs = size(q_Bathory(n_Bathory+1:end)); 
        qDs = size(q_Darkthrone(n_Darkthrone+1:end)); 
        qMs = size(q_Mayhem(n_Mayhem+1:end)); 
        positive2 = [sum(pre2(1:qBs(2))==1); sum(pre2(qBs(2)+1:qBs(2)+qDs(2))==2); sum(pre2(qBs(2)+qDs(2)+1:end)==3)]; 
        Positive2 = [Positive2, positive2]; 
    end 
    clearvars -except spec2 Positive2 qBs qDs qMs pre2 
    
    accuracy2 =[mean(Positive2(1,:))/qBs(2), mean(Positive2(2,:))/qDs(2), mean(Positive2(3,:))/qMs(2)]; 
    accuracy2 
    
    figure()
    bar(accuracy2,'k')
    title('Test 2 Classification Accuracy','fontsize',30)
end 


%% Genre Classification
function Question2_Part3 
    n = 5; % sample size 
    
    % Electronic: Deadmau5, Daft Punk, David Guetta
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Electro\1_Faxing_Berlin.mp3');
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Electro\2_Daft_Punk_ATW.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Electro\3_David_Guetta_Memories.mp3'); 
    [spec11]= mus2spect(y11, fs11, n); 
    [spec12]= mus2spect(y12, fs12, n); 
    [spec13]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Classical: Mozart, Bach, Beethoven
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Classical\1_Eine_Kleine_Nachtmusik.mp3'); 
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Classical\2_Air_Bach.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Classical\3_Fur_Elise.mp3'); 
    [spec14]= mus2spect(y11, fs11, n); 
    [spec15]= mus2spect(y12, fs12, n); 
    [spec16]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 

    % Metal: Bathory, Metallica, Slayer 
    [y11, fs11]= audioread('C:\Users\skewl\Desktop\Music\Metal\1_A_Fine_Day_to_Die.mp3'); 
    [y12, fs12]= audioread('C:\Users\skewl\Desktop\Music\Metal\2_For_Whom_The_Bell_Tolls.mp3'); 
    [y13, fs13]= audioread('C:\Users\skewl\Desktop\Music\Metal\3_Slayer_Raining_Blood.mp3');
    [spec17]= mus2spect(y11, fs11, n); 
    [spec18]= mus2spect(y12, fs12, n); 
    [spec19]= mus2spect(y13, fs13, n); 
    clearvars y11 fs11 y12 fs12 y13 fs13 
    
    % SVD 
    Electro_size = size([spec11, spec12, spec13]); 
    Classical_size = size([spec14, spec15, spec16]); 
    Metal_size = size([spec17, spec18, spec19]); 
    spec1 = [spec11, spec12, spec13, spec14, spec15, spec16, spec17, spec18, spec19]; 
    clearvars -except spec1 Electro_size Classical_size Metal_size 
 
    [u3,s3,v3] = svd(spec1,'econ'); 
    diags1 = diag(s3)/sum(diag(s3)); 

    figure()
    plot(diags1,'k*','Linewidth',2)
    title('Test 3 Singular Values','fontsize',30);
    xlabel('Singular Value','fontsize',30), ylabel('Fractional Value','fontsize',30)
    grid on 

    figure()
    plot(1:length(s3), 0.85*ones(length(s3)),'r-','Linewidth',2)
    hold on 
    
    percent3 = zeros(length(s3),1); 
    for i = 1:length(s3)
        percent3(i) = sum(diags1(1:i)); 
    end
    
    plot(1:length(s3), percent3,'k*'), hold on 
    title('Test 3 Modes vs % Reconstruction','fontsize',30); 
    xlabel('Modes','fontsize',30), ylabel('% Reconstruction','fontsize',30)
    grid on 
    
    percent = 0.1:0.1:0.9; 
    coln = []; 
    for i = 1:length(percent)
        [row,col] = find(percent3>percent(i)); 
        coln = [coln,col(1)];
    end
    
    % Classification
    Positive3 = []; 
    for i = 1:1000
        nt = 0.5; % for train data 
        nmin = coln(2); % reconstruction lower bound 
        nmax = coln(8); % upper bound 
        n_Electro = round(Electro_size(2)*nt); 
        n_Classical = round(Classical_size(2)*nt); 
        n_Metal = round(Metal_size(2)*nt); 
        q_Electro = randperm(Electro_size(2)); 
        q_Classical = randperm(Classical_size(2)); 
        q_Metal = randperm(Metal_size(2)); 
        x_Electro = v3(1: Electro_size(2),:);
        x_Classical = v3(Electro_size(2)+1: Electro_size(2)+Classical_size(2),:);
        x_Metal = v3(Electro_size(2)+Classical_size(2)+1: Electro_size(2)+Classical_size(2)+Metal_size(2),:); 
        xtrain3 = [x_Electro(q_Electro(1: n_Electro),:); x_Classical(q_Classical(1: n_Classical),:); x_Metal(q_Metal(1:n_Metal),:)]; 
        xtest3 = [x_Electro(q_Electro(n_Electro+1:end),:); x_Classical(q_Classical(n_Classical+1:end),:); x_Metal(q_Metal(n_Metal+1:end),:)]; 
        ctrain3 = [ones(n_Electro,1); 2*ones(n_Classical,1); 3*ones(n_Metal,1)]; 
        xtrain3 = real(xtrain3); % Train data
        xtest3 = real(xtest3); % Test data
        ctrain3 = real(ctrain3); % Label train data
        nb3 = fitcnb(xtrain3, ctrain3); % Naive Bayes
        pre3 = nb3.predict(xtest3); % Prediction
        q_Electros = size(q_Electro(n_Electro+1:end)); 
        q_Classicals = size(q_Classical(n_Classical+1:end)); 
        q_Metals = size(q_Metal(n_Metal+1:end)); 
        positive3 = [sum(pre3(1:q_Electros(2))==1); sum(pre3(q_Electros(2)+1:q_Electros(2)+q_Classicals(2))==2); sum(pre3(q_Electros(2)+q_Classicals(2)+1:end)==3)];
        Positive3 = [Positive3, positive3]; 
    end 
    clearvars -except v1 spec1 Electro_size Classical_size Metal_size Positive3 q_Electros q_Classicals q_Metals pre1 
    
    accuracy3 = [mean(Positive3(1,:))/q_Electros(2), mean(Positive3(2,:))/q_Classicals(2), mean(Positive3(3,:))/q_Metals(2)]; 
    accuracy3 

    figure()
    bar(accuracy3,'k')
    title('Test 3 Classification Accuracy','fontsize',30)
end


%% Read Audioread data to Spectrogram
function[spect] = mus2spect(y,fs,n)
    y = sum(y,2)/2; 
    sampling = y(1:n:end); 
    sample5stack =[];
    for j = 1:round(length(sampling)/(fs/n)/5)-1 
        sample5 = sampling(1+5*(j-1)*fs/n:5*j*fs/n); 
        sample5stack = [sample5stack,sample5]; 
    end 
    spect = [];  
    for j = 1:round(length(sampling)/(fs/n)/5)-1 
        spect5 = spectrogram(sample5stack(:,j),fs/n); 
        spect = [spect,spect5(:)]; 
    end 
end