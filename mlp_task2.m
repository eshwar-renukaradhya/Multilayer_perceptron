clc;
clear;
close all;

train = load('mnist_train.csv');
images = train(:,2:785);
labels = train(:,1);
labels_new = [];
num = [];
% load 100 images for each number(1000 images)
i = 1;
cnt=1;
for j = 0:9
    while cnt<=100
        if(labels(i,1)==j)
            num = [num ; images(i,:)];
            labels_new = [labels_new ; j];
            cnt = cnt+1;
        end
        i=i+1;
    end
    i=1;
    cnt =1;
end

%% normalization
X = [];
M = 10;
P = 1000;
num = num/255;
%shuffling
ordered =[labels_new num];
shuffle = ordered(randperm(size(ordered,1)),:);
num = shuffle(:,2:785);
labels_new = shuffle(:,1);
X=num';


% Desired values
desired=zeros(M,P);

for idx=1:P
    
    desired(labels_new(idx,1)+1,idx)=1;
    
end

%% Training
% initialize
N=784;

no=[];
error = [];
eta = 0.003;
L=[15 20 30];


% iterate to each value of L
for L_idx=1:length(L)
    
    weights=randn(L(L_idx),M);% Random values for window weights
    v_wt = randn(N,L(L_idx));
    u=zeros(L(L_idx),1);
    y=zeros(M,P);
    
    for k=1:200 % Convergence
        
        for S=1:P % Each Image
        
            Net_k = v_wt'*X(:,S);
            

            % activation function

            den=1+exp(-Net_k);

            a=1./den;

            u = a;  
            
            Net_j = weights'*u;
            
                
            % activation function

            den_y=1+exp(-Net_j);
              
            a_y=1./den_y;
                
            y(:,S) = a_y;
              
            % update weights
                
            delta_j(:,S) = (desired(:,S) - y(:,S));
            delta_w = eta*u*(delta_j(:,S) .* y(:,S) .*(1-y(:,S)))';
            
            delta_k = weights*delta_j(:,S);
            delta_v = eta*X(:,S)*(delta_k .* u .* (1-u))';
            
            weights = weights + delta_w;
            v_wt = v_wt + delta_v;
            
        end
        
        % Mean square error
        error_num=sum(delta_j(:).^2);
        error_den=M*P;
        error(L_idx,k)=error_num/2;
        no(L_idx,k)=k;
        
        
    end
    % store weights for each L value
    if(L_idx == 1)
        wt1 = weights;
        v_wt1 = v_wt;
    elseif(L_idx == 2)
        wt2 = weights;
        v_wt2 = v_wt;
    else
        wt3 = weights;
        v_wt3 = v_wt;
    end
end

%% plot linear curve
figure;
subplot(2,3,1);
plot(no(1,:),error(1,:),'b','LineWidth',2);
xlabel(' No. of iterations');
ylabel (' Mean square Error');
title('Different L - 1000 Training Images');
legend('L=15');

subplot(2,3,2);
plot((no(2,:)),error(2,:),'r','LineWidth',2);
xlabel(' No. of iterations');
ylabel (' Mean square Error');
title('Different L - 1000 Training Images');
legend('L=20');

subplot(2,3,3);
plot((no(3,:)),error(3,:),'k','LineWidth',2);
xlabel(' No. of iterations');
ylabel (' Mean square Error');
title('Different L - 1000 Training Images');
legend('L=30');
%% testing
% load test data
test = load('mnist_test.csv');
images_t = test(:,2:785);
labels_t = test(:,1);
i = 1;
cnt=1;
v=[];
num = [];
% load 30 images for each number(300 images)
for j = 0:9
    while cnt<=30
        if(labels_t(i,1)==j)
            num = [num ; images_t(i,:)];
            cnt = cnt+1;
        end
        i=i+1;
    end
    i=1;
    cnt =1;
end
num = num';

for L_idx=1:length(L)

    if(L_idx == 1)
        weights = wt1;
        v_wt = v_wt1;
    elseif(L_idx == 2)
        weights = wt2;
        v_wt = v_wt2;
    else
        weights = wt3;
        v_wt = v_wt3;
    end
    
    Net_k_t = v_wt'*num;
    u_t = 1./(1+exp(-Net_k_t));
    Net_j_t = weights'*u_t;
    v = 1./(1+exp(-Net_j_t));
    
        
    % count the total number of each digit
    values = [];
    for S=1:300
        [val , idx] = max(v(:,S));
        values = [values ; idx-1];
    end
    count = 0;
    val_cnt=[];
    for j = 0:9
        for S = 1:300
            if(values(S,1)==j)
                count = count+1;
            end
        end
        val_cnt = [val_cnt count];
        count=0;
    end
    
    % plot percentage error
    subplot(2,3,3+ L_idx)
    bar(abs(1-val_cnt/30));
    xlabel(' Numbers 0 to 9');
    ylabel (' Percentage Error');
    title('Percentage error');
end

