classdef baum_welch_functions
    methods
        function [new_A ,new_B ,new_c , LogLik ] = baum_welch_multiobs_norm (obj, A, B, O, c)
        % Baum - Welch algorithm : for multiobservations and with normalization
        % [A ,B ,c , LogLik ]= baum_welch_multiobs(A, B, O, c)
        % m hidden states , n output states and N observations
        % A - mxm (state transitions matrix)
        % B - nxm (confusion matrix)
        % O - 1xN (observations vector)
        % c - 1xm (priors vector)
        % LogLik - log likelihood of the observation sequence O
        [route_nbr ,~]= size(O) ;
        GamaT =0;
        tauT =0;
        tauiT =0;
        OmegaT =0;
        nuT =0;
        LogLik =0;
            
            for route =1: route_nbr ,
                [ Alfa , LP, ~] = forward_algorithm_norm(A, B, O(route, :), c);
                LogLik = LogLik + LP ;
                Beta = backward_algorithm_norm(A, B, O(route, :));
                Gama = obj.compute_gama (Alfa, Beta);
                GamaT = GamaT + Gama ;
                tau = obj.compute_tau(Alfa, Beta, A, B, O(route, :));
                tauT = tauT + tau;
                taui = obj.compute_taui(Gama, B, O(route, :));
                tauiT = tauiT + taui;
                nu = obj.compute_nu(Gama, B);
                nuT = nuT + nu;
                Omega = obj.compute_omega(Gama, B, O(route, :));
                OmegaT = OmegaT + Omega;
            end
            
            new_c = GamaT(1, :) / route_nbr;
            new_A = tauT ./ tauiT;
            new_B = OmegaT ./ nuT;
            
        end
        function [new_A ,new_B ,new_c ]= baum_welch_algorithm (obj, A ,B ,O ,c )
        % Baum - Welch algorithm
        % m hidden states , n output states and N observations
        % A - mxm ( state transitions matrix )
        % B - nxm ( confusion matrix )
        % O - 1 xN ( observations vector )
        % c - 1 xm ( priors vector )
            Alfa = forward_algorithm(A, B, O, c) ;
            Beta = backward_algorithm(A, B, O);
            Gama = obj.compute_gama(Alfa, Beta);
            tau = obj.compute_tau(Alfa, Beta, A, B, O) ;
            taui = obj.compute_taui(Gama, B , O) ;
            nu = obj.compute_nu(Gama, B);
            Omega = obj.compute_omega(Gama, B, O);
            new_c = Gama(1 ,:) ;
            new_A = tau ./ taui ;
            new_B = Omega ./ nu ;

            varargout(1) ={new_A }; 
            varargout(2) ={new_B};
            varargout(3) ={new_c};
        end

        function nu = compute_nu(obj, Gama ,B)
        % Return the number of visits to state i
        % m hidden states , n output states and N observations
        %
        % B - m , n ( confusion matrix )
        % nu - mxn matrix
            [~ , n ]= size (B);
            nu =(sum(Gama)).' * ones(1, n); % Sum along the columns of Gamma
        end
        
        function taui = compute_taui (obj, Gama ,B ,O )
        % Compute taui expected number of transition made from si
        % m hidden states , n output states and N observations
        %
        % Gama - Nxm ( from the forward algorithm )
        % O - 1 xN ( observations vector )
        % nu - Return an mxm matrix
            [m ,~]= size(B );
            N= length(O);
            taui = Gama(1: N -1 ,:) ;
            taui =(sum(taui,1) ).' * ones(1 , m );
        end

        function tau = compute_tau(obj, Alfa, Beta, A, B, O)
        % Compute tau -- estimated transition frequency between pair of states
        % (expected transitions)
        % m hidden states , n output states and N observations
        % Alfa - Nxm ( from the forward algorithm )
        % Beta - Nxm ( from the backward algorithm )
        % A - mxm ( state transitions matrix )
        % B - nxm ( confusion matrix )
        % O - 1 xN ( observations vector )
        %  Ratio taui/tau will be estimation for transition matrix
        %  Ratio between expected number of transitions from s_i to s_j and total
        %  number of transition from s_i
            [m, ~]= size (B );
            N = length(O);
            tau = zeros(m, m) ;
            for k = 1: N -1 ,
                num = A .* ( Alfa(k, :).' * Beta(k +1, :) ) .* (B(:, O(k +1) ) * ones(1, m)).';
                den = ones(1, m) * num * ones(m, 1) ;
                tau = tau + num / den ;
            end
        end


        function Gama = compute_gama(obj, Alfa, Beta )
        % Compute gamma_k (i) - the probability of being at state i at step k given
        % sequence
        % Alfa - Nxm ( from the forward algorithm )
        % Beta - Nxm ( from the backward algorithm )
        % Gama - Return an Nxm matrix with the shape :
        %    _                           _
        %   | gama_1 (1) ... gama_1 ( m ) |
        %   | ...                     ... |
        %   | gama_N (1) ... gama_N ( m ) |
        %    -                           -
        %   gamma1_(i) will be initial probabilities estimate. and part of emission
        %   probs estimate
            [~ , m ]= size ( Alfa );
            P= diag ( Alfa * Beta') * ones (1 , m);
            Gama =(Alfa .* Beta ) ./ P ;
        end
        
        function Omega = compute_omega(obj, Gama, B, O)
        % Compute omega - number of times that being in state s_i we have seen r_j
        % m hidden states , n output states and N observations
        % Gama - Nxm matrix
        % B - nxm ( confusion matrix )
        % O - 1 xN ( observations vector )
        % this will be top in emission matrix estimation 
        % "ratio between transitions expected value for observation r_j and total
        % number of visits to state s_i"
        [m , n ]= size (B );
            for j =1: n ,
                inx = find(O == j);
                if ~ isempty(inx) ,
                    Omega(:, j )= sum(Gama(inx, :), 1).' ;
                else
                    Omega (:, j ) = 0 * ones(m, 1) ;
                end
            end
        end
    end
end



















