clear all ;
clc ;
m =2;
n =3;
O= generate_data () ;
MaxIter = 100;
[A ,B ,c , Fit ] = hmm_rosen(m, n, O, MaxIter);


plot (1:100 , Fit (:) );
xlabel ('Iteration');
ylabel ('Log - Likelihood');
grid on ;
figure