function O = generate_data ()
    % Function used to generate training data from the HMM
    % with parameters :
    c =[0.6;0.4];
    A =[0.7 0.3;0.4 0.6];
    B =[0.1 0.4 0.5;0.7 0.2 0.1];

    % Data generation considering route_nbr different routes
    % and observ_nbr observations per route
    route_nbr =10;
    observ_nbr =2000;

    % Initialize matrices :
    H = zeros(route_nbr, observ_nbr ); % Hidden
    O = zeros(route_nbr, observ_nbr); % Observable

    for route =1: route_nbr ,
        H(route ,1) = random_func(c);
        for p = 2 : observ_nbr ,
            H(route, p) = random_func (A( H( route ,p -1) ,:) );
        end
        for p = 1: observ_nbr ,
            O(route ,p) = random_func (B(H(route, p),:)) ;
        end
    end
end


function roulette = random_func ( prob , N)
% N - Number of elements
% prob - Pr ob ab il it ie s distribution : vector 1 xn with distinct
% probability distribution
% roulette - Vector 1 xN with values within 1 and n according
% to distribution prob
    if nargin ==1 ,
        N = 1;
    end
    % roulette wheel where each slice is proportional to
    % the probability
    L= length ( prob );
    prob = round (100* prob ) ;
    % roulette - vector with 100 elements whose distribution
    % depends on P . For example if P =[0.3 0.7] then :
    % roulette = [1 1 ... 1 2 2 ...2...2]
    % \ - - - - - - -/ \ - - - - - - - - - -/
    %       30                  70
    roulette =[];
    for k =1 : L ,
        roulette =[roulette k * ones(1 , prob (k)) ];
    end

    % Generates N values evenly distributed between 1 and 100
    % ( it will be the index of the " roulette " vector )
    ptr = round(99* rand (1 , N) +1) ;
    roulette = roulette( ptr ) ;
end