

% State transition matrix (N x N)

A_init = [0.5, 0.5 ;
          0.5, 0.5;];

% Emission probabilities (each row with sum 1)
B_init = [0.25, 0.25, 0.25, 0.25;
          0.10, 0.40, 0.40, 0.10;];

% Observed sequency
%ACGT
O= [2, 3, 1, 2, 4, 1, 2, 4, 3, 1, 2, 4, 1, 2, 4, 2, 3, 2, 2, 3, 1, 2, 3, 2, 3, 1, 2, 4, 3, 2, 2, 3, 4, 2, 4, 1, 4, 1, 2, 4, 3, 2, 3, 2, 1, 4, 1, 2, 3, 3, 2];

% Initial state occupance probabilities
I_init = zeros(size(A_init,1),1); 
I_init(1) = 0.5;
I_init(2) = 0.5;



[Alfa, P_f] = forward_algorithm(A_init,B_init,O,I_init);

[Beta] = backward_algorithm(A_init,B_init,O);


[q] = viterbi_algorithm(A_init, B_init, O, I_init);



bw_obj = baum_welch_functions;

[new_A, new_B, new_c] = bw_obj.baum_welch_algorithm (A_init ,B_init ,O ,I_init );



Alfa = forward_algorithm(A_init,B_init,O,I_init);
Beta = backward_algorithm(A_init,B_init,O);
Gama = bw_obj.compute_gama(Alfa, Beta);


% normalized forward algorithm
[Alfa_norm, norm_P_f, u]= forward_algorithm_norm(A_init ,B_init ,O ,I_init );
[Beta_norm_old] = backward_algorithm_norm(A_init,B_init,O);
[Beta_norm_new] = backward_algorithm_norm(new_A,new_B,O);

% Observed sequency
%ACGT
O= [2, 3, 1, 2, 4, 1, 2, 4, 3, 1, 2, 4, 1, 2, 4, 2, 3, 2, 2, 3, 1, 2, 3, 2, 3, 1, 2, 4, 3, 2, 2, 3, 4;
    2, 3, 1, 2, 4, 1, 2, 4, 3, 1, 2, 4, 1, 2, 4, 2, 3, 2, 1, 4, 1, 2, 4, 3, 2, 3, 2, 1, 4, 1, 2, 3, 3 ];

[new_A, new_B, new_c, log_prob] = bw_obj.baum_welch_multiobs_norm(A_init ,B_init ,O ,I_init);


%% Viterbi training
O = [2, 3, 1, 2, 4, 1, 2, 4, 3, 1, 2, 4, 1, 2, 4, 2, 3, 2, 2, 3, 1, 2, 3, 2, 3, 1, 2, 4, 3, 2, 2, 3, 4 ];
MaxIter = 10;

m = 2;
n = 4;

[A ,B ,c , Fit ] = viterbi_training(m ,n ,O , MaxIter);



[ALPHA, OMEGA] = gradientLA(A_init ,B_init ,O ,I_init);
[ALPHA_norm, OMEGA_norm] = gradientLA_norm(A_init ,B_init, O, I_init);

[ALPHA_B, OMEGA_B] = gradientLB(A_init ,B_init ,O ,I_init);

[ALPHA_B_norm, OMEGA_B_norm] = gradientLB_norm(A_init ,B_init, O, I_init);

[c , dLdc] = gradientLC(A_init ,B_init ,O ,I_init);