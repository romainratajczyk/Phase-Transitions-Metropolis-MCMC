#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono> 
#include <omp.h>
#include <fftw3.h>
#include <deque>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// incorporer nouveau warmup.

const double SQRT3 = std::sqrt(3.0);

// reciprocal lattice, rescaling, pure ferromagnet. Pas de cooling. voir presentation du code rapide article 2 (mts)


inline int pbc(int x, int L) { return (x + L) % L; }



// J1 : 6 voisins (1er anneau) distance 1a (hexagon)
const int tri_neighbors[6][2] = {
    {0,+1}, {+1,0}, {+1, -1},
    {0, -1}, {-1, 0}, {-1, +1}
};

// J2 : 6 voisins (2ème anneau) distance sqrt(3)a
const int tri_J2_neighbors[6][2] = {
    {-1,+2}, {+1, +1}, {+2, -1},
    {+1, -2}, {-1, -1}, {-2, 1}
};

// J3 : 6 voisins (3ème anneau) distance 2a
const int tri_J3_neighbors[6][2] = {
    {0, +2}, {+2, 0}, {+2, -2},
    {0, -2}, {-2, 0}, {-2, +2}
};




/* Remark : The triangular geometry is not encoded in the real-space positions of the spins, 
but in the way we define the neighbor lists. We start from the spin at (i,j) and apply the offsets to reach the neighbors. 
The triangular geometry is then entirely defined by the J1,J2 and J3 neighbors if they are correctly defined, mentally on an odd line, spins are at the real position (i,j+1/2) 
(we should keep that in mind)
*/

// energie locale pour Metropolis
double delta_energy(
    const std::vector<std::vector<double>>& S,
    int i, int j,
    double phi_new,
    double J1,
    double J2,
    double J3
) {
    int L = S.size();
    double phi_old = S[i][j];
    double dE = 0.0;
    for (auto& o : tri_neighbors) { // o fait référence à la collection de tri_neighbors
        int ni = pbc(i + o[0], L), nj = pbc(j + o[1], L);
        double phin = S[ni][nj];
        dE -= J1 * (cos(phi_new - phin) - cos(phi_old - phin));
    }
    for (auto& o : tri_J2_neighbors) {
        int ni = pbc(i + o[0], L), nj = pbc(j + o[1], L);
        double phin = S[ni][nj];
        dE -= J2 * (cos(phi_new - phin) - cos(phi_old - phin));
    }
    for (auto& o : tri_J3_neighbors) {
        int ni = pbc(i + o[0], L), nj = pbc(j + o[1], L);
        double phin = S[ni][nj];
        dE -= J3 * (cos(phi_new - phin) - cos(phi_old - phin));
    }
    return dE; // pas de double comptage ici, c'est un calcul local pour Metropolis
}

// Computation of total energy (not the local energy - which is used for Metropolis)
double compute_energy(const std::vector<std::vector<double>>& S, double J1, double J2, double J3) {
    int L = S.size();
    double E = 0.0;
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            double phi = S[x][y];
            for (auto& o : tri_neighbors) {
                int nx = pbc(x + o[0], L), ny = pbc(y + o[1], L);
                E += -J1 * cos(phi - S[nx][ny]);
            }
        }
    }
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            double phi = S[x][y];
            for (auto& o : tri_J2_neighbors) {
                int nx = pbc(x + o[0], L), ny = pbc(y + o[1], L);
                E += -J2 * cos(phi - S[nx][ny]);
            }
        }
    }
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            double phi = S[x][y];
            for (auto& o : tri_J3_neighbors) {
                int nx = pbc(x + o[0], L), ny = pbc(y + o[1], L);
                E += -J3 * cos(phi - S[nx][ny]);
            }
        }
    }
    return E / 2.0; // pour éviter simplement le double counting
}

//Effective field (pour overrelaxation et choix de l'angle pour metropolis)
void effective_field(
    const std::vector<std::vector<double>>& S,
    int i, int j,
    double J1, double J2, double J3,
    double& hx, double& hy
) {
    int L = S.size();
    hx = 0.0; hy = 0.0;
    // J1
    for (auto& d : tri_neighbors) {
        double ph = S[pbc(i + d[0],L)][pbc(j + d[1],L)];
        hx += J1*cos(ph); hy += J1*sin(ph);
    }
    //J2
    for (auto& d : tri_J2_neighbors) {
        double ph = S[pbc(i + d[0],L)][pbc(j + d[1],L)];
        hx += J2*cos(ph); hy += J2*sin(ph);
    }
    //J3
    for (auto& d : tri_J3_neighbors) {
        double ph = S[pbc(i + d[0],L)][pbc(j + d[1],L)];
        hx += J3*cos(ph); hy += J3*sin(ph);
    }
}

// Order parameter O 
std::complex<double> compute_order_parameter_O(const std::vector<std::vector<double>>& S, int L) {
    const std::complex<double> omega = std::polar(1.0, 2.0 * M_PI / 3.0);
    const std::complex<double> omega2 = std::polar(1.0, 4.0 * M_PI / 3.0);

    std::complex<double> O_sum = 0.0;
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            double theta = S[x][y];
            // a2, a1, puis a3 dans l'ordre 
            double c1 = cos(theta - S[pbc(x + tri_neighbors[0][0], L)][pbc(y + tri_neighbors[0][1], L)]);
            double c2 = cos(theta - S[pbc(x + tri_neighbors[1][0], L)][pbc(y + tri_neighbors[1][1], L)]);
            double c3 = cos(theta - S[pbc(x + tri_neighbors[5][0], L)][pbc(y + tri_neighbors[5][1], L)]); 
            O_sum += 0.5 * (c1 + omega * c2 + omega2 * c3);
        }
    }
    return O_sum / (double)(L * L);
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    const int L       = 150 ;                 
    const long long WARMUP   =  1000000LL * L * L / 3 ;   
    const long long N_trials = 250000LL * L * L  ;    
    const double k = 1.0 ;     
    const double J1    = 1.0; 
    double delta =  0.20   ; 
    //if (argc > 2) delta = atof(argv[2]);
    const double J2 = - ( delta + 0.20 ) ;       // J > 0 ===> ferromagnetic 
    //const double J2    = 1.0 * -J1 / 2.0;
    const double J3    = 1.0 * J2 / 2.0 ;          // take care of dividing by 2.0 not 2 which is an integer division
    const double p_over = 2.0/3.0;           // probability for overrelaxation move to occur
    const int measurements_spacing = 5 * L * L;   // compute energy etc every " " steps
    /* Remarks : not a great difference between WARMUP=3000, N_trials = 50 000 (L=50) 
    and WARMUP = 3000, N_trials = 100 000 (L=100). First one even seems look better */
    // Au début de main, avant les phases :
    std::deque<double> last_ratios1;
    std::deque<double> ratio_window;
    std::deque<double> sector_window;
    std::mt19937 rng(std::time(nullptr));
    std::uniform_real_distribution<double> unif(0.0,1.0);
    std::uniform_real_distribution<double> angle_dist(0.0, 2*M_PI);
    std::uniform_int_distribution<int> site_dist(0, L-1);

    std::vector<double> T_list; // for each temeprature, the code performs a warmup phase and a measurement phase
    for (double T = 0.5; T > 0.4 - 1e-8; T -= 0.005) T_list.push_back(T);
    //for (double T = 0.005; T > 0.001 - 1e-8; T -= 0.001) T_list.push_back(T); 
    //for (double T = 0.0009; T > 0.0001 - 1e-8; T -= 0.0001) T_list.push_back(T); 
   
    int idx = 0;
    if (argc > 1) idx = atoi(argv[1]);
    if (idx < 0 || idx >= (int)T_list.size()) {
        std::cerr << "Indice de température invalide !" << std::endl;
        return 1;
    }
    double T = T_list[idx];

    /*std::ostringstream fname;
    fname << "xy_triangular_O"
          << "_L_" << L
          << "_delta_" << delta
          << "_J2_" << J2
          << "_1_J3_" << J3
            << "_McWARMUP_" << WARMUP / ( L*L)
          << "_McTRIALS_" << N_trials / ( L*L)
          << "_T_" << T
          << ".dat"; // indiquer tous les paramètres pour chaque simulation 
    std::ofstream fout(fname.str());
    fout << "# T E_incr M O C chi\n" << std::fixed << std::setprecision(6); */
    std::ostringstream fname;
    fname << "xy_triangular_O"
          << "_L_" << L
          << "_delta_" << delta
          << "_J2_" << J2
          << "_1_J3_" << J3
            << "_McWARMUP_" << WARMUP / ( L*L)
          << "_McTRIALS_" << N_trials / ( L*L)
          << "_T_" << T
          << ".dat"; // indiquer tous les paramètres pour chaque simulation 
    std::ofstream fout(fname.str());
    fout << "# T O \n" << std::fixed << std::setprecision(6);
 
    double beta = 1.0 / T; 


    // Réseau triangulaire : vecteurs de base
const double a1[2] = {1.0, 0.0};
const double a2[2] = {0.5, SQRT3/2.0};

// Reciprocal lattice vectors
const double b1[2] = {2*M_PI, -2*M_PI/SQRT3};
const double b2[2] = {0.0, 4*M_PI/SQRT3};


    // vecteurs qui vont contenir pour chaque voisins de 0 à 5, les distances réelles 
std::pair<double, double> tri_neighbors_real[6];
std::pair<double, double> tri_J2_neighbors_real[6];
std::pair<double, double> tri_J3_neighbors_real[6];

// calcul des distances réelles (besoin que de scalar product)
for (int k = 0; k < 6; ++k) {
    // Voisins J1
    int dx1 = tri_neighbors[k][0];
    int dy1 = tri_neighbors[k][1];
    tri_neighbors_real[k] = std::make_pair(dx1 * a1[0] + dy1 * a2[0],
                                           dx1 * a1[1] + dy1 * a2[1]);

    // Voisins J2
    int dx2 = tri_J2_neighbors[k][0];
    int dy2 = tri_J2_neighbors[k][1];
    tri_J2_neighbors_real[k] = std::make_pair(dx2 * a1[0] + dy2 * a2[0],
                                              dx2 * a1[1] + dy2 * a2[1]);

    // Voisins J3
    int dx3 = tri_J3_neighbors[k][0];
    int dy3 = tri_J3_neighbors[k][1];
    tri_J3_neighbors_real[k] = std::make_pair(dx3 * a1[0] + dy3 * a2[0],
                                              dx3 * a1[1] + dy3 * a2[1]);
}



// Hot-start: random initialization
    std::vector<std::vector<double>> S(L, std::vector<double>(L));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            S[i][j] = angle_dist(rng);
    double E_total = compute_energy(S, J1, J2, J3); 
    

// Précalcule les positions réelles pour chaque site (x, y)
std::vector<std::vector<double>> rx_tab(L, std::vector<double>(L));
std::vector<std::vector<double>> ry_tab(L, std::vector<double>(L));
for (int x = 0; x < L; ++x)
    for (int y = 0; y < L; ++y) {
        rx_tab[x][y] = x * a1[0] + y * a2[0];
        ry_tab[x][y] = x * a1[1] + y * a2[1];
    }




/* --- Warmup phases ---
    double sector_angle = M_PI ; // initial sector angle (sector width is 2*sector_angle, the sector is [-sector_angle, sector_angle])
    double target_acceptance = 0.5;
    double K = 1.0; // à ajuster dynamiquement near 50% acceptance zone
    long long warmup1 = WARMUP / 5; // first little thermalisation phase
    long long warmup2 = 2 * WARMUP / 5; // second phase, 40% of warmup, to adapt the sector width. Take into account that the sector width is adjusted every monte carlo step.
    // So for a warmup phase of 3000 * L^2 steps, the sector angle is adjusted 1200 times
    long long warmup3 = WARMUP - warmup1 - warmup2; // third phase, 40% of warmup too, final thermalisation without adaption of the sector width to avoid interferences */ 





    // --- WARMUP PHASES (copie fidèle de warmup_test.cpp) ---

    // --- Cooling schedule ---
    const int N_cool = 8;
    const double alpha = 3.0;
    std::vector<double> T_cool_list;
    double T_cool_start = T + 0.5 ; 
    double T_cool_end = T;
    bool is_lowT_mode = (T <= 0.1);
    double p_global_init = is_lowT_mode ? 0.02 : 0.0;
    double p_global_final = 0.0;
    double p_over_special = is_lowT_mode ? 0.70 : p_over;
    double K0 = 1.0; 
    double sector_angle = M_PI;           // largeur initiale du secteur
    double target_acceptance = 0.5;

    if (T > 0.1) {
        // inverse log-cooling for high temperatures
        for (int k = 0; k < N_cool; ++k) {
            double x = double(k) / (N_cool - 1);
            double T_cool = T_cool_end + (T_cool_start - T_cool_end) * std::pow(1.0 - x, alpha);
            T_cool_list.push_back(T_cool);
        }
    } else {
        // geometric cooling for low temperatures
        for (int k = 0; k < N_cool; ++k) {
            double r = double(k) / (N_cool - 1);
            double T_cool = T_cool_start * std::pow(T_cool_end / T_cool_start, r);
            T_cool_list.push_back(T_cool);
        }
    }

    size_t n_cool = T_cool_list.size();
    long long step_global = 0;

    // --- Warmup progression ---
    double warmup_fraction_start = 0.25;
    double warmup_fraction_end = 1.0;

   

    for (size_t t_idx = 0; t_idx < n_cool; ++t_idx) {
        double T_cool = T_cool_list[t_idx];
        beta = 1.0 / T_cool;

        // Progression linéaire du warmup_fraction
        double warmup_fraction = warmup_fraction_start +
            (warmup_fraction_end - warmup_fraction_start) *
            (T_cool_start - T_cool) / (T_cool_start - T_cool_end);
        long long WARMUP_cool = (long long)(WARMUP * warmup_fraction);

        long long warmup1 = WARMUP_cool / 5;
        long long warmup2 = 2 * WARMUP_cool / 5;
        long long warmup3 = WARMUP_cool - warmup1 - warmup2;

        // --- Phase 1 ---
        long long n_accept1 = 0, n_total1 = 0;
        double sigma = is_lowT_mode ? 0.75 * sector_angle : sector_angle / 2.0;
        std::normal_distribution<double> gauss(0.0, sigma);
        std::cout << "Start of Phase 1 for T_cool = " << T_cool << std::endl << std::flush;
        for (long long step = 0; step < warmup1; ++step, ++step_global) {
            int i = site_dist(rng), j = site_dist(rng);
            if (unif(rng) < p_over_special) {
                double hx, hy;
                effective_field(S, i, j, J1, J2, J3, hx, hy);
                double theta_eff = atan2(hy, hx);
                S[i][j] = fmod(2*theta_eff - S[i][j] + 2*M_PI, 2*M_PI);
            } else {
                double phi_old = S[i][j];
                double phi_new;
                if (unif(rng) < p_global_init) {
                    phi_new = angle_dist(rng);
                } else {
                    
                    phi_new = phi_old + gauss(rng);
                }
                double dE = delta_energy(S, i, j, phi_new, J1, J2, J3);
                if (dE <= 0 || unif(rng) < exp(-beta * dE)) {
                    S[i][j] = phi_new;
                    S[i][j] = fmod(S[i][j] + 2*M_PI, 2*M_PI);
                    E_total += dE;
                    n_accept1++;
                }
                n_total1++;
            }
            if (step_global % (100 * L * L) == 0 && n_total1 > 0) {
                double ratio = double(n_accept1) / n_total1;
                last_ratios1.push_back(ratio);
                if (last_ratios1.size() > 100) last_ratios1.pop_front();
                n_accept1 = 0; n_total1 = 0;
            }
        }

        // --- Phase 2 (adaptation) ---
        const int adapt_interval = L * L;
        const int window_size = 20;
        const int sector_window_size = 1000;
        long long n_accept2 = 0, n_total2 = 0;
        long long n_adapt = 0;
        std::cout << "Start of Phase 2 for T_cool = " << T_cool << std::endl << std::flush;
        // Adaptation initiale basée sur la moyenne des derniers ratios de phase 1
        double ratio1_avg = 0.0;
        for (double r : last_ratios1) ratio1_avg += r;
        if (!last_ratios1.empty()) ratio1_avg /= last_ratios1.size();
        if (!last_ratios1.empty()) {
            double K = K0;
            sector_angle *= exp(K * (ratio1_avg - target_acceptance));
            if (sector_angle > M_PI) sector_angle = M_PI;
            if (sector_angle < 0.01) sector_angle = 0.01;
            sector_window.push_back(sector_angle);
        }

        for (long long step = 0; step < warmup2; ++step, ++step_global) {
            int i = site_dist(rng), j = site_dist(rng);
            if (unif(rng) < p_over_special) {
                double hx, hy;
                effective_field(S, i, j, J1, J2, J3, hx, hy);
                double theta_eff = atan2(hy, hx);
                S[i][j] = fmod(2*theta_eff - S[i][j] + 2*M_PI, 2*M_PI);
            } else {
                double phi_old = S[i][j];
                double phi_new;
                if (unif(rng) < p_global_init) {
                    phi_new = angle_dist(rng);
                } else {
                    sigma = is_lowT_mode ? 0.75 * sector_angle : sector_angle / 2.0;
                    gauss = std::normal_distribution<double>(0.0, sigma);
                    phi_new = phi_old + gauss(rng); // nouveau move dans phi_old + quelque part dans l'intervalle centré en zéro avec une certaine proba
                }
                double dE = delta_energy(S, i, j, phi_new, J1, J2, J3);
                if (dE <= 0 || unif(rng) < exp(-beta * dE)) {
                    S[i][j] = phi_new;
                    S[i][j] = fmod(S[i][j] + 2*M_PI, 2*M_PI);
                    E_total += dE;
                    n_accept2++;
                }
                n_total2++;
            }
            if ((step+1) % adapt_interval == 0 && n_total2 > 0) {
                double ratio = double(n_accept2) / n_total2;
                n_adapt++;
                ratio_window.push_back(ratio);
                if (ratio_window.size() > window_size)
                    ratio_window.pop_front();
                double ratio_avg = 0.0;
                for (double r : ratio_window) ratio_avg += r;
                ratio_avg /= ratio_window.size();
                double K = K0 / std::sqrt(n_adapt);
                sector_angle *= exp(K * (ratio_avg - target_acceptance));
                if (sector_angle > M_PI) sector_angle = M_PI;
                if (sector_angle < 0.01) sector_angle = 0.01;
                sector_window.push_back(sector_angle);
                if (sector_window.size() > sector_window_size)
                    sector_window.pop_front();
                n_accept2 = 0; n_total2 = 0;
            }
        }

        // --- Phase 3 ---
        double sector_angle_sum = 0.0;
        std::cout << "Start of Phase 3 for T_cool = " << T_cool << std::endl << std::flush;
        for (double s : sector_window) sector_angle_sum += s;
        if (!sector_window.empty())
            sector_angle = sector_angle_sum / sector_window.size();
        sigma = is_lowT_mode ? 0.75 * sector_angle : sector_angle / 2.0;
        gauss = std::normal_distribution<double>(0.0, sigma);

        long long n_accept3 = 0, n_total3 = 0;
        for (long long step = 0; step < warmup3; ++step, ++step_global) {
            double p_global = p_global_init * (1.0 - double(step) / warmup3);
            int i = site_dist(rng), j = site_dist(rng);
            if (unif(rng) < p_over_special) {
                double hx, hy;
                effective_field(S, i, j, J1, J2, J3, hx, hy);
                double theta_eff = atan2(hy, hx);
                S[i][j] = fmod(2*theta_eff - S[i][j] + 2*M_PI, 2*M_PI);
            } else {
                double phi_old = S[i][j];
                double phi_new;
                if (unif(rng) < p_global) {
                    phi_new = angle_dist(rng);
                } else {
                    
                    phi_new = phi_old + gauss(rng);
                }
                double dE = delta_energy(S, i, j, phi_new, J1, J2, J3);
                if (dE <= 0 || unif(rng) < exp(-beta * dE)) {
                    S[i][j] = phi_new;
                    S[i][j] = fmod(S[i][j] + 2*M_PI, 2*M_PI);
                    E_total += dE;
                    n_accept3++;
                }
                n_total3++;
            }
        }
    }
    // --- Fin du warmup, S et E_total sont prêts pour la phase de mesure ---
    // Measurement phase (sector is fixed at each temperature, according to the warmup phase)
    double sumE = 0, sumE2 = 0;
    double sumM = 0, sumM2 = 0;
    double sumU = 0;
    std::complex<double> sumO = 0.0;
    const int Nq = L; // size of reciprocal lattice 
    static std::vector<std::vector<double>> Sq(Nq, std::vector<double>(Nq, 0.0));
    long long Sq_count = 0; 
    long long count = 0;

    fftw_complex* inx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L * L);
    fftw_complex* iny = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L * L);
    fftw_complex* outx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L * L);
    fftw_complex* outy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L * L);
    fftw_plan planx = fftw_plan_dft_2d(L, L, inx, outx, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plany = fftw_plan_dft_2d(L, L, iny, outy, FFTW_FORWARD, FFTW_ESTIMATE); 
    
    double sigma = sector_angle / 2.0;
    std::normal_distribution<double> gauss(0.0, sigma);
    for (long long step = 0; step < N_trials; ++step) {
        int i = site_dist(rng), j = site_dist(rng);
        if (unif(rng) < p_over) {
            double hx, hy;
            effective_field(S, i, j, J1,J2, J3, hx, hy);
            double theta_eff = atan2(hy, hx);
            S[i][j] = fmod(2*theta_eff - S[i][j] + 2*M_PI, 2*M_PI);
            
        } else {
            double phi_old = S[i][j];
            double phi_new = phi_old + gauss(rng);
            double dE = delta_energy(S, i, j, phi_new, J1,J2, J3);
            if (dE <= 0 || unif(rng) < exp(-beta * dE)) {
                S[i][j] = phi_new;
                S[i][j] = fmod(S[i][j] + 2*M_PI, 2*M_PI);
                E_total += dE;
            }
        }

       
        

        if (step % measurements_spacing == 0) { // call the fonction compute_energy and calculate magnetization, variances every measurement_spacing steps
            

        // Calcul du structure factor S(q) avec positions réelles. Donnera l'info pour chaque T, si il y a une phase ordonnée préférée
        // Nq is the size of the reciprocal lattice 
       auto t0 = std::chrono::high_resolution_clock::now();

        for (int x = 0; x < L; ++x)
            for (int y = 0; y < L; ++y) {
            double phi = S[x][y];
            inx[x*L + y][0] = cos(phi);
            inx[x*L + y][1] = 0.0;
            iny[x*L + y][0] = sin(phi);
            iny[x*L + y][1] = 0.0;
            }

        fftw_execute(planx);
        fftw_execute(plany);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "FFT duration: " << std::chrono::duration<double>(t1-t0).count() << " s" << std::endl;

    // Remplir Sq (attention à la normalisation et au centrage)
    #pragma omp parallel for collapse(2)
    for (int im = 0; im < L; ++im) {
        for (int in = 0; in < L; ++in) {
            int qm = (im + L/2) % L;
            int qn = (in + L/2) % L;
            double re_x = outx[qm*L + qn][0], im_x = outx[qm*L + qn][1];
            double re_y = outy[qm*L + qn][0], im_y = outy[qm*L + qn][1];
            double Sq_val = (re_x*re_x + im_x*im_x + re_y*re_y + im_y*im_y) / (L*L);
            Sq[im][in] += Sq_val;
        }
    }
    Sq_count++;  




        /*    // helicity modulus calculation (using real distances)
            // Calcul du module d'hélicité direction x
               double helicity1_x = 0.0, helicity2_x = 0.0;
            for (int x = 0; x < L; ++x) {
                for (int y = 0; y < L; ++y) { // scan all the lattice sites 
                    double phi = S[x][y];
                    // J1
                    for (int k = 0; k < 6; ++k) {
                        double dr_x = tri_neighbors_real[k].first; // x component of the real vector which link to the neighbor
                        if (dr_x != 0.0) { // take into account only the neighbors which contribute to the helicity modulus in the x direction
                            int nx = pbc(x + tri_neighbors[k][0], L); // neighbors position with periodic boundary conditions
                            int ny = pbc(y + tri_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J1 * dr_x; // J1 times the scalar product (e_ij is a unit vector in the direction of the neighbor, tricky point. Should it appear ?  )
                            helicity1_x += weight * dr_x * cos(dphi);
                            helicity2_x += weight * sin(-dphi); // two terms for the helicity modulus 
                        }
                    }
                    // J2
                    for (int k = 0; k < 6; ++k) {                    
                        double dr_x = tri_J2_neighbors_real[k].first;
                        if (dr_x != 0.0) {
                            int nx = pbc(x + tri_J2_neighbors[k][0], L);
                            int ny = pbc(y + tri_J2_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J2 * dr_x;
                            helicity1_x += weight * dr_x * cos(dphi);
                            helicity2_x += weight * sin(-dphi);
                        }
                    }
                    // J3
                    for (int k = 0; k < 6; ++k) {
                        double dr_x = tri_J3_neighbors_real[k].first;
                        if (dr_x != 0.0) {
                            int nx = pbc(x + tri_J3_neighbors[k][0], L);
                            int ny = pbc(y + tri_J3_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J3 * dr_x;
                            helicity1_x += weight * dr_x * cos(dphi);
                            helicity2_x += weight * sin(-dphi);
                        }
                    }
                }
            }
            helicity1_x /= (2.0 * L * L); // check the sum in the articles : here we scan all the neighbors of i for each i site. in the article it seems they scan all j sites for all i sites
            helicity2_x /= (2.0 * L ); // L will be squared just int the following line
            double U_x = helicity1_x - beta * helicity2_x * helicity2_x;

            // Calcul du module d'hélicité direction y
            double helicity1_y = 0.0, helicity2_y = 0.0;
            for (int x = 0; x < L; ++x) {
                for (int y = 0; y < L; ++y) {
                    double phi = S[x][y];
                    // J1
                    for (int k = 0; k < 6; ++k) {
                        double dr_y = tri_neighbors_real[k].second;
                        if (dr_y != 0.0) {
                            int nx = pbc(x + tri_neighbors[k][0], L);
                            int ny = pbc(y + tri_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J1 * dr_y;
                            helicity1_y += weight * dr_y * cos(dphi);
                            helicity2_y += weight * sin(-dphi);
                        }
                    }
                    // J2
                    for (int k = 0; k < 6; ++k) {
                        double dr_y = tri_J2_neighbors_real[k].second;
                        if (dr_y != 0.0) {
                            int nx = pbc(x + tri_J2_neighbors[k][0], L);
                            int ny = pbc(y + tri_J2_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J2 * dr_y;
                            helicity1_y += weight * dr_y * cos(dphi);
                            helicity2_y += weight * sin(-dphi);
                        }
                    }
                    // J3
                    for (int k = 0; k < 6; ++k) {
                        double dr_y = tri_J3_neighbors_real[k].second;
                        if (dr_y != 0.0) {
                            int nx = pbc(x + tri_J3_neighbors[k][0], L);
                            int ny = pbc(y + tri_J3_neighbors[k][1], L);
                            double dphi = phi - S[nx][ny];
                            double weight = J3 * dr_y;
                            helicity1_y += weight * dr_y * cos(dphi);
                            helicity2_y += weight * sin(-dphi);
                        }
                    }
                }
            }
            helicity1_y /= (2.0 * L * L);
            helicity2_y /= (2.0 * L );
            double U_y = helicity1_y - beta * helicity2_y * helicity2_y;

            // Moyenne des deux directions
            double U_avg_dir = (U_x + U_y) / 2.0;
            sumU += U_avg_dir; */

            // Calcul du paramètre d'ordre nématique
            std::complex<double> O = compute_order_parameter_O(S, L);
            sumO += O;
            
            //double E_incr = E_total;
            

            /*double mx = 0, my = 0;
            for (int x = 0; x < L; ++x)
                for (int y = 0; y < L; ++y) {
                    mx += cos(S[x][y]);
                    my += sin(S[x][y]);
                }
            mx /= (L * L); // checker si pas de pb de normalization (check: OK)
            my /= (L * L);
            double M = sqrt(mx*mx + my*my);

            sumE  += E_incr / (L*L);
            sumE2 += (E_incr / (L*L)) * (E_incr / (L*L));
            
            sumM  += M;
            sumM2 += M*M; */
            ++count;    
            // Affichage de la progression toutes les 5% (par exemple)
            if (step % (N_trials / 20) == 0) {
                double percent = 100.0 * step / N_trials;
                std::cout << "Progress: " << std::fixed << std::setprecision(1)
                          << percent << "% (" << step << "/" << N_trials << " steps)" << std::endl;
            }
        }
    }
    // compute E/N, C_v, <|M|>/N, chi (critical moment for normalizations)
    /*double e_avg  = sumE  / count; // average of energy per site 
    double e2_avg = sumE2 / count;
    double m_avg  = sumM  / count; // count ou double(count) c'est pareil car le numérateur est un double
    double m2_avg = sumM2 / count;
    //double U_avg = sumU / count;*/
    std::complex<double> O_avg = sumO / double(count); 
    double O_mod = std::abs(O_avg);                    // Module de O
    double O_arg = std::arg(O_avg);                    // Argument de O 
    
    /*double C   = beta*beta * (e2_avg - e_avg*e_avg) * (L*L);
    double chi = beta       * (m2_avg - m_avg*m_avg) * (L*L);*/

    /*fout << T << " "
         << e_avg << " "
         //<< U_avg << " "
         << m_avg << " "
         << O_mod << " "
          
         << C     << " "
         << chi   << "\n"; */
    fout << T << " " << O_mod << "\n";
        
    

    fout.close(); 

    // Save the structure factor S(q) to a separated file 
    // --- Structure factor S(q) = |Sx(q)|^2 + |Sy(q)|^2, centré sur la zone de Brillouin ---. Voir note cahier
    std::ostringstream fname_sq;
    fname_sq << "measure_Sq_O"
             << "_L_" << L
             << "_delta_" << delta
             << "_J2_" << J2
             << "_1_J3_" << J3
             << " _McWARMUP_" << WARMUP / ( L*L)
             << "_McTRIALS_" << N_trials / ( L*L)
             << "_T_" << T
             << ".dat";
    std::ofstream fout_sq(fname_sq.str());
    for (int im = 0; im < L; ++im) {
        int m = im - L/2;
        for (int in = 0; in < L; ++in) {
            int n = in - L/2;
            double qx = (m * b1[0] + n * b2[0]) / L;
            double qy = (m * b1[1] + n * b2[1]) / L;
            double Sq_val = Sq[im][in] / double(Sq_count); // MOYENNE sur toutes les mesures, pas la derniere mesure comme warmup_test.cpp
            fout_sq << qx << " " << qy << " " << Sq_val << "\n";
        }
        fout_sq << "\n";
    }
    
    fout_sq.close();   

    

    // Sauvegarde la configuration finale des spins
    std::ostringstream fname_conf;
    fname_conf << "spins_config_O"
               << "_L_" << L
               << "_delta_" << delta
               << "_J2_" << J2
               << "_1_J3_" << J3
               << "_McWARMUP_" << WARMUP / ( L*L)
               << "_McTRIALS_" << N_trials / ( L*L)
               << "_T_" << T
               << ".dat";
    std::ofstream fout_conf(fname_conf.str());
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            fout_conf << S[x][y] << (y == L-1 ? "\n" : " ");
        }
    }
    fout_conf.close();

    auto end = std::chrono::high_resolution_clock::now(); //fin
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total running duration : " << elapsed.count() << " seconds." << std::endl;
    
    
    
    
    fftw_destroy_plan(planx);
    fftw_destroy_plan(plany);
    fftw_free(inx); fftw_free(iny); fftw_free(outx); fftw_free(outy); 

    

    return 0;
}







// to run :  g++ -std=c++11 XY_triangular.cpp -o XY_triangulars && ./XY_triangular 

/* étapes pour envoyer sur cluster :  ADAPTER LE NOM DES FICHIERS 

ssh ratajczyk@curta.zedat.fu-berlin.de

cd xy_project


scp /Users/romain/Desktop/C++/XY_triangular.cpp ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/ 
sur MAC, entrer MDP.

puis compiler avec 

g++ -O3 -march=native -funroll-loops -ffast-math -fopenmp -std=c++11 -o XY_triangular XY_triangular.cpp -lm -lfftw3

le job est : "

#!/bin/bash
#SBATCH --array=0-14
#SBATCH --job-name=xy_T%a
#SBATCH --output=xy_out_%A_%a.txt
#SBATCH --error=xy_err_%A_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

./XYmodel_three_coupling_constants $SLURM_ARRAY_TASK_ID

", l'envoyer avec 

sbatch job_xy.slurm (nano job_xy.slurm pour le lire)

Puis checker l'avancée avec 

squeue -u ratajczyk

Et récupérer les résultats avec 
scp "ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/xy_triangular_BKT_L_90_T_*.dat" ~/Desktop/C++/results_bests_measures/

suivre : 

tail xy_T_1.5.dat

ARRETER le job : 

scancel 23115060

 
scp "ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/xy_triangular_O_L_150_*" ~/Desktop/C++/RESULTS/FINAL_CODE/MEASURES/Observables/heatmap_O6
scp "ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/xy_Tscan_out_23654898_*.txt" ~/Desktop/C++/RESULTS/FINAL_CODE/MEASURES/Average_Sq/RAPPORT/heatmap_Sq_O6
scp "ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/spins_config_O_L_150_delta_0.07_*" ~/Desktop/C++/RESULTS/FINAL_CODE/MEASURES/Spins_config/rapport



voir les activités des derniers jobs : 
sacct -u ratajczyk --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End




#!/bin/bash
#SBATCH --array=0-4999%700
#SBATCH --job-name=xy_%a
#SBATCH --output=xy_out_%A_%a.txt
#SBATCH --error=xy_err_%A_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=10MB
#SBATCH --time=20:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

export OMP_NUM_THREADS=1

# Offset pour continuer la numérotation logique
offset=5001
real_id=$((SLURM_ARRAY_TASK_ID + offset))
k=$((real_id / 114))
idx=$((real_id % 114))
delta=$(echo "0.0025 * $k" | bc -l)
./XY_triangular $idx $delta


array 0-10 000 pour le moment. Offset pour finir ensuite


23493655_1055 (2e lancé avec offset) a été tué (posait problème) 


rsync -av --min-size=1 'ratajczyk@curta.zedat.fu-berlin.de:~/xy_project/xy_triangular_O_L_150_*' ~/Desktop/C++/RESULTS/FINAL_CODE/MEASURES/Observables/heatmap_O6/

-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_0.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_10.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_11.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_12.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_13.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_14.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_15.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_16.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_17.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 17:01  xy_Tscan_err_23654898_18.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 17:01  xy_Tscan_err_23654898_19.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_1.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_2.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_3.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_4.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_5.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_6.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_7.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:56  xy_Tscan_err_23654898_8.txt
-rw-r----- 1 ratajczyk agreuther   99 23 juil. 16:58  xy_Tscan_err_23654898_9.txt
-rw-r----- 1 ratajczyk agreuther  291 23 juil. 16:53  xy_Tscan_out_23654898_0.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:47  xy_Tscan_out_23654898_10.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:38  xy_Tscan_out_23654898_11.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:56  xy_Tscan_out_23654898_12.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:56  xy_Tscan_out_23654898_13.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:55  xy_Tscan_out_23654898_14.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:42  xy_Tscan_out_23654898_15.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:36  xy_Tscan_out_23654898_16.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:43  xy_Tscan_out_23654898_17.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:57  xy_Tscan_out_23654898_18.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:56  xy_Tscan_out_23654898_19.txt
-rw-r----- 1 ratajczyk agreuther  264 23 juil. 16:40  xy_Tscan_out_23654898_1.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:53  xy_Tscan_out_23654898_2.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:51  xy_Tscan_out_23654898_3.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:46  xy_Tscan_out_23654898_4.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:32  xy_Tscan_out_23654898_5.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:34  xy_Tscan_out_23654898_6.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:49  xy_Tscan_out_23654898_7.txt
-rw-r----- 1 ratajczyk agreuther  300 23 juil. 16:42  xy_Tscan_out_23654898_8.txt
-rw-r----- 1 ratajczyk agreuther  303 23 juil. 16:47  xy_Tscan_out_23654898_9.txt

*/

