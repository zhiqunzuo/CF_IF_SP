data {
    int<lower = 0> N; // number of observation
    int<lower = 0> K; // number of covariates
    matrix[N, K]   a; // sensitive variables
    array[N] real  ugpa; //UGPA
    array[N] real   lsat; //LSAT
    real           ugpa0;
    real           eta_u_ugpa;
    vector[K]      eta_a_ugpa;
    real           lsat0;
    real           eta_u_lsat;
    vector[K]      eta_a_lsat;
}

parameters {
    vector[N] u;
}

model {
    u ~ uniform(-2, 2);

    //ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, 0.00001);
    //lsat ~ normal(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat), 0.00001);
    ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, 0.1);
    lsat ~ normal(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat), 0.1);
}