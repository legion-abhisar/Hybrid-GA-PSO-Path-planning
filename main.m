clc;
close all;

CostFunction = @(x) Sphere(x);%Cost Function

nVar=10;            % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size
VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables

MaxIt=100;      % Maximum Number of Iterations
nPop=100;       % Population Size (Swarm Size)

% PSO Parameters -->
w=1;                    % Inertia Weight
wdamp=0.99;             % Inertia Weight Damping Ratio
c1=1.5;                 % Personal Learning Coefficient
c2=2.0;                 % Global Learning Coefficient
pc=0.7;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (also Parnets)
gamma=0.4;              % Extra Range Factor for Crossover
pm=0.3;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants
mu=0.1;                 % Mutation Rate
beta=8;

GlobalBest.Cost=inf;  % Initialize Global Best
pop=repmat(empty_individual,nPop,1);  % Create Particles Matrix

%Initialization-->

% Create Empty Particle Structure
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

%particle(i) = zeros(1,1000000);  %preallocating

for i = 1:nPop
    % Initialize Position
    pop(i).Position = unifrnd(VarMin,VarMax,VarSize);
    
    % Evaluation
    pop(i).Cost = CostFunction(pop(i).Position);
    particle(i).Position = pop(i).Position;
    
    % Initialize Velocity
    particle(i).Velocity = zeros(VarSize);
    
    % Evaluation
    particle(i).Cost = pop(i).Cost;
    
    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost        
        GlobalBest = particle(i).Best;
    end
end

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

% Store Cost
WorstCost=pop(end).Cost;

Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

for it=1:MaxIt
    
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        % Select Parents Indices
        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position,gamma,VarMin,VarMax);
        
        % Evaluate Offsprings
        popc(k,1).Cost=CostFunction(popc(k,1).Position);
        popc(k,2).Cost=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu,VarMin,VarMax);
        
        % Evaluate Mutant
        popm(k).Cost=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop
        popc
        popm]; %#ok
    
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    
    
    % Store Best Cost Ever Found
    
    for i=1:nPop
        if pop(i).Cost<=particle(i).Cost
            particle(i).Position = pop(i).Position;
            particle(i).Cost = pop(i).Cost;
        end
        Cx(i) = particle(i).Cost;
    end
    [BestCost(it),r]=min(Cx);
    GlobalBest.Cost=particle(r).Cost;
    GlobalBest.Position=particle(r).Position;
    
    BstCostGA(it)=BestCost(it);
    for i=1:nPop
        
        
        
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    for i=1:nPop
        if particle(i).Cost<=pop(i).Cost
            pop(i).Position = particle(i).Position;
            pop(i).Cost = particle(i).Cost;
            
        end
        Cx(i)=pop(i).Cost;
    end
    
    [BestCost(it),r]=min(Cx);
    GlobalBest.Cost=pop(r).Cost;
    GlobalBest.Position=pop(r).Position;
    
    BstCostPSO(it)=BestCost(it);
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    w=w*wdamp;
end

BestSol = GlobalBest;

figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;