function [A ,B ,c , Fit ]= viterbi_training (m ,n ,O , MaxIter )
% Viterbi learning algorithm
% m hidden states , 
% n output states and 
% N observations
% O - 1 x N ( observations vector )
% MaxIter - maximum number of iterations

    N = length (O);
    
    % Random PARAMETERS INITIALIZATION
    A = rand (m ,m); 
    A = A ./ (sum(A').' * ones(1, m));
    B = rand(m, n); 
    B = B ./ ( sum(B').' * ones(1 , n));
    c = rand(m , 1) ; 
    c = c / sum(c);
    
    % ITERATION
    for k =1: MaxIter ,

        % Determines performance of new parameters ...
        [pd ,P ] = forward_algorithm_norm(A, B, O, c);
        disp ([ 'Iteration -- ' num2str(k) ' FIT -- ' num2str(P)]);
        Fit(k)=P;
    
        % SEGMENTATION : VITERBI ALGORITHM
        Q = viterbi_algorithm (A, B, O, c).';
    
        % ESTIMATION c
        c = zeros(2 ,1) ;
        c(Q(1)) =1;
    
        % ESTIMATION A
        for i =1: m ,
            for j =1: m ,
                Num =(sum((i * ones(1, N - 1) == Q(1 : N - 1)) .* (j * ones(1, N - 1) == Q(2 : N))));
                Den =(sum((i * ones(1 ,N - 1) == Q(1 : N - 1)))) ;
                if Den ~=0 ,
                    A(i, j) = Num / Den ;
                else
                    A(i, j) = 1;
                end
            end
        end

        % ESTIMATION B
        for i =1: m ,
            for j =1: n ,
                k = find(O == j) ;
                if isempty(k) ,
                    B(i, j) = 0;
                else
                    Num = (sum((i * ones(1, length(k)) == Q(k))));
                    Den = (sum((i * ones(1, N) == Q( 1 : N)))) ;
                    if Den ~= 0 ,
                        B(i, j) = Num / Den ;
                    else
                        B(i, j) = 1;
                    end
                end
            end
        end
    end
end


