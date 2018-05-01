%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main file to test normalization of probabilistic
%inference in a lower dimensional space represented
%by a series of Gaussians.

%An even distribution of Gaussians over -1,1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Constants
num_basis = 20; %Number of basis functions in representation
range = [-1 1]; %Represented range
numsteps = 300; %Number of steps on the range
var = .01; %variance of Gaussians

%Calculate more constants from above
dx = (range(2)-range(1))/(numsteps-1); %Step size over range
x = range(1):dx:range(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Choose a basis set and its inverse
if 1 == 0 %Do a random representation picked from an even dist
    x_mean = 0;
else %evenly space the means on the range.
    dmean = (range(2)-range(1))/(num_basis+1);
    x_mean = range(1)+dmean:dmean:range(2)-dmean;
%     dmean = (range(2)-range(1)-32*var)/(num_basis+1);
%     x_mean = range(1)+16*var+dmean:dmean:range(2)-dmean-16*var;
end
basis = exp(-(x'*ones(1,num_basis)-ones(numsteps,1)*x_mean).^2/var);
basis = basis./(sqrt(sum(basis.^2))'*ones(1,numsteps))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the pinv of the basis for bias calculations
decoders = pinv(basis);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the Probabilistic inference matrix i.e., p(a|b)
% p(a) = int(p(a|b) p(b) db)
%
%This can be anything that has the same thing
%in every row (shifted version), where rows are p(a|b=B)

%Gaussians along the diagonal (slowly (by var) moves input to even dist)
P = exp(-(x'*ones(1,numsteps)-ones(numsteps,1)*x).^2/(5*var));

%An even distribution (rapidly (1 step) moves input to even)
%P = ones(numsteps,numsteps);

%An Identity matrix perserves the input
%P = diag(ones(1,numsteps));

%Normalize the chosen p(a|b) along rows
P = P./(dx*sum(P,2)*ones(1,numsteps));  %sum(P')*dx = ones(1,numsteps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pick the starting distribution for p(a)
%p_a = ones(1,numsteps)/(dx*numsteps); %even dist over the range, area = 1
p_a = exp(-(x-.0).^2/(1*var)); %Normal distribution in middle
p_a = p_a/(dx*sum(p_a)); %Normalize it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Put the dist into the basis space
p_a1 = basis'*p_a';
p_a = p_a';
p_a2 = zeros(100+1,num_basis);
p_a2(1,:) = p_a1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Perform the normalization correction
%calculate bias directly
bias = decoders'*basis'*ones(numsteps,1);
P_bias = P./(ones(numsteps,1)*bias');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Probabilistic inference over and over with 
%and without bias added
ideal_area = 0; bias_area = 0; nobias_area=0;
figure(5);clf;hold on;
for i = 1:100
    %ideal (no encoding/decoding)
    p_a = P*p_a*dx;
    ideal_area(i) = sum(p_a)*dx;
    %without bias
    p_a1 = basis'*(P*decoders')*p_a1*dx;
    nobias_area(i)=sum(decoders'*p_a1)*dx;
    %with bias
    p_a2(i+1,:) = basis'*(P_bias*decoders')*p_a2(i,:)'*dx;
    bias_area(i)=sum(decoders'*p_a2(i+1,:)')*dx;
    
    if i==10 || i==100 || i==5 %|| i==1
        plot(x,p_a,'-','LineWidth',2,'Color',[.5 .5 .5]); 
        plot(x,decoders'*p_a2(i+1,:)','k','LineWidth',1);
        plot(x,decoders'*p_a1,'k--','LineWidth',1);
    end
end

%Labels for the time course plot in the previous loop
figure(5); hold on;
xlabel('x')
ylabel('\rho(x)')
legend('ideal','with bias','without bias');
title('Comparison of ideal, biased, and unbiased inference over time');
% figure(1);clf;hold on;
% plot(p_a1);
% plot(p_a2,'r:');

%Plot of the basis functions used
figure(2); clf;
plot(x,basis,'k');
title('Neuron-like basis functions');
xlabel('x'); ylabel('Activity');

%Plot of the decoder functions
figure(3); clf; hold on;
plot(x,decoders(5,:),'k:');
plot(x,decoders(10,:),'k');
title('Decoding functions for neuron-like encoders');
xlabel('x'); ylabel('d(x)');

%The conditional inference distribution
figure(10); clf; colormap('gray');
mesh(P);
title('P(y|x)'); xlabel('x');ylabel('y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot the change in area in the decoded space
%over time with and without bias
figure(4);clf;hold on;
plot(ideal_area,'-','LineWidth',2,'Color',[.5 .5 .5]);
plot(bias_area,'k','LineWidth',1);
plot(nobias_area,'k--','LineWidth',1);
title('Bias Affect on Density Area');
legend('ideal','with bias','without bias');
xlabel('Iteration'); ylabel('Area');


