// Single-file structural QAIA decoder for the v3 manifest.
// dSB/bSB check-product dynamics, at most six trajectories, fifty steps.
// Only QAIA check products are used; no external decoding stages.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using std::size_t;

static constexpr int STEPS = 50;
static constexpr double EPS = 1e-12;

enum class Mode { Dsb, Bsb, KSpin };
enum class Combine { Energy, Vote, SoftVote };

struct TrajectoryConfig {
    double tau;
    double xi;
    double lambd;
    double delta;
    double energy_lambda;
    double dt;
    const double* tau_values;
    const double* xi_values;
    const double* lambd_values;
    const double* delta_values;
    const double* dt_values;
    Mode mode;
    double beta;          // Onsager/TAP reaction strength (bSB only); 0 disables
    double parity_scale;  // check-force scale on parity columns (last m); 1 disables
    const double* beta_values;
    int joint_pairs;      // 1 = exact Gray-pair channel field (bSB only)
    double pair_gamma;    // pair reaction strength
    double pair_scale;    // diagnostic scale on the pair coupling; 1 is exact
    double g;             // KSpin parity-coupling scale
    double clip;          // KSpin soft-clip radius for parity LLRs
    int select_unclipped; // KSpin energy selection uses raw parity LLRs
};

struct CaseConfig {
    int n;
    int k;
    double snr;
    Combine combine;
    Mode mode;
    int count;
    const TrajectoryConfig* trajectories;
};

static const double TRAJ_0_0_XI[STEPS] = {0.39601500903053155, 0.39602082599201621, 0.39601239823565926, 0.39601777686262501, 0.39613880502406545, 0.3966494351596685, 0.39617522986069847, 0.39600828793081067, 0.3960249732700678, 0.39601981646289719, 0.39602005986667543, 0.39601692133678423, 0.39600667812761486, 0.39600992078754377, 0.39600837275000161, 0.39601605290195468, 0.39602179628179113, 0.39600873283517246, 0.39600680180976566, 0.39600986196657501, 0.39600960291296816, 0.39601076775311422, 0.39601210135957837, 0.39606223924733636, 0.39602506957352035, 0.39601814171550237, 0.39602289367906279, 0.39601475585310975, 0.39601609699015666, 0.39602241376393232, 0.3960269765101026, 0.39604171618374318, 0.39605384614111883, 0.39608083307456621, 0.39614239583283128, 0.39619592515194502, 0.39627631547520459, 0.39647024205084008, 0.39678225646208315, 0.39712327058356228, 0.39735467714207451, 0.39748645334816257, 0.39793885370836313, 0.39853085318674786, 0.39873439780698239, 0.39942708138111865, 0.40002590161060714, 0.40061603537576096, 0.40089898012895708, 0.40074670881839425};
static const double TRAJ_0_0_LAMBD[STEPS] = {5.9401697209969369, 5.9404763297702319, 5.9403565752840262, 5.9400975248002892, 5.9405215430781375, 5.9403101283503146, 5.9401664413812245, 5.940153528964343, 5.9403240448195582, 5.9403311296608452, 5.9403383462152242, 5.9402854658785795, 5.9401411872123102, 5.9401988844478337, 5.940169619372246, 5.9401694980727413, 5.9401761861795741, 5.9401149019864796, 5.9400855154144505, 5.9400999625395858, 5.9400995102721943, 5.9401174333633273, 5.9400725319415058, 5.940109080440175, 5.940094783991964, 5.9401617319192654, 5.9402042992990918, 5.9401323181769143, 5.9401117886941313, 5.9402312352349682, 5.9402550936899106, 5.9403435162179248, 5.9403519084159173, 5.9404900279575479, 5.9407251583963161, 5.9409771536466423, 5.9413556789013509, 5.9420749116195877, 5.9431908033935992, 5.9446325399652027, 5.9466245605614079, 5.9496738037188805, 5.9552280071297474, 5.9640203825149909, 5.9717373666650859, 5.9847512936577747, 5.9949510122918159, 6.0074563883568608, 6.0124474838272031, 6.0106528145026878};
static const double TRAJ_0_0_DELTA[STEPS] = {2.0201132256379437, 2.020169296964089, 2.0201851432201967, 2.0201490244374312, 2.0201510247297838, 2.0201426069487636, 2.0201283763188416, 2.0200562907926023, 2.0200802423123898, 2.0200393105986816, 2.0201049658102512, 2.0201570303511276, 2.0201555565492, 2.0201608097843939, 2.0201721112636819, 2.0201805311434069, 2.0201816756122386, 2.0201845671328016, 2.0201847601176186, 2.0201854735042235, 2.0201847581642585, 2.020178570630955, 2.0201667336295102, 2.0201408881940837, 2.0201240209840585, 2.0201406817616805, 2.0201344049819121, 2.0201333401497146, 2.0201311047606136, 2.020061340063636, 2.0200244459511589, 2.0200158053925263, 2.0199444664366402, 2.0198475476588222, 2.0196195165505828, 2.0193940725360178, 2.0188140145615137, 2.0180071856938016, 2.0171828177261455, 2.01710828144902, 2.0167627460426805, 2.0158642640756717, 2.0147383619683126, 2.0134120764619539, 2.0122378233721077, 2.0109056587571539, 2.0096143083628033, 2.0081917793733304, 2.0061394402006658, 2.003623509414854};
static const double TRAJ_0_0_DT[STEPS] = {0.19800598129670416, 0.1980098663927502, 0.19800946642439782, 0.19800281894324892, 0.19801638695245444, 0.19801127048256001, 0.19802030086624789, 0.19802679204301865, 0.19800298657810081, 0.19800930941920428, 0.19801725818333127, 0.19803084146532204, 0.20060852505360185, 0.19803774656842327, 0.19805544887472593, 0.20053214893080537, 0.20081862698127084, 0.19970267089543473, 0.20175479677136671, 0.20197760551036031, 0.20193607600658575, 0.19994698781831874, 0.20065442451402643, 0.20196854891506696, 0.20189813378975438, 0.20084651129607314, 0.20143699400802947, 0.20130527837470316, 0.20094653593118755, 0.19823605486998583, 0.19820216623370129, 0.19806132487695347, 0.19806571520738811, 0.1981114517375166, 0.19811690542067603, 0.1981392278520053, 0.19814275334070339, 0.19820711172402736, 0.19832075215321177, 0.19857869062820926, 0.19905067882206381, 0.1991379534080735, 0.19963124625370107, 0.20018819193871262, 0.20024863843914742, 0.20053048656396585, 0.20061274123661302, 0.20072495300004603, 0.2008239373488975, 0.20088813891844046};
static const TrajectoryConfig TRAJ_0[6] = {{8.0803580992176549, 0.39601500903053155, 5.9401697209969369, 2.0201132256379437, 6, 0.19800598129670416, nullptr, TRAJ_0_0_XI, TRAJ_0_0_LAMBD, TRAJ_0_0_DELTA, TRAJ_0_0_DT, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}, {4, 0.5, 6, 2.5, 6, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}, {4, 0.5, 5, 2.5, 5, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}, {4, 0.40000000000000002, 5, 2, 5, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}, {4, 0.40000000000000002, 6, 2, 6, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}, {8, 0.5, 6, 2.5, 6, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Dsb, 0, 1, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const CaseConfig FALLBACK = {0, 0, 0, Combine::Energy, Mode::Dsb, 6, TRAJ_0};

static const TrajectoryConfig TRAJ_1[1] = {{2.976, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0, 1, 0.97999999999999998, 2.7999999999999998, 0}};

static const TrajectoryConfig TRAJ_2[1] = {{5.1840000000000002, 0.59999999999999998, 9.375, 3, 9.375, 0.40000000000000002, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.16, 0.63749999999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_3[1] = {{5, 0.40000000000000002, 10, 2.25, 10, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.12, 0.87187499999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_4[1] = {{2.976, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0, 1, 0.97999999999999998, 2.7999999999999998, 0}};

static const TrajectoryConfig TRAJ_5[1] = {{5.1840000000000002, 0.59999999999999998, 9.375, 3, 9.375, 0.40000000000000002, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.16, 0.63749999999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_6[1] = {{5, 0.40000000000000002, 10, 2.25, 10, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.12, 0.87187499999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_7[3] = {{2.976, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0, 1, 0.97999999999999998, 2.7999999999999998, 0}, {3.5, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0, 1, 0.97999999999999998, 2.7999999999999998, 0}, {2.976, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0.10000000000000001, 1, 0.97999999999999998, 2.7999999999999998, 0}};

static const TrajectoryConfig TRAJ_8[1] = {{5.1840000000000002, 0.59999999999999998, 9.375, 3, 9.375, 0.40000000000000002, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.16, 0.63749999999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_9[1] = {{5, 0.40000000000000002, 10, 2.25, 10, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.12, 0.87187499999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_10[1] = {{2.976, 1, 1, 3.4409999999999998, 1, 0.46999999999999997, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::KSpin, 0.317, 1, nullptr, 1, 0, 1, 0.97999999999999998, 2.7999999999999998, 0}};

static const TrajectoryConfig TRAJ_11[1] = {{5.1840000000000002, 0.59999999999999998, 9.375, 3, 9.375, 0.40000000000000002, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.16, 0.63749999999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const TrajectoryConfig TRAJ_12[1] = {{5, 0.40000000000000002, 10, 2.25, 10, 0.20000000000000001, nullptr, nullptr, nullptr, nullptr, nullptr, Mode::Bsb, 0.12, 0.87187499999999996, nullptr, 0, 0, 1, 1, 1e+30, 0}};

static const CaseConfig CASES[12] = {{648, 324, 6, Combine::Energy, Mode::KSpin, 1, TRAJ_1}, {648, 432, 9, Combine::Energy, Mode::Bsb, 1, TRAJ_2}, {648, 486, 10.5, Combine::Energy, Mode::Bsb, 1, TRAJ_3}, {1296, 648, 6, Combine::Energy, Mode::KSpin, 1, TRAJ_4}, {1296, 864, 9, Combine::Energy, Mode::Bsb, 1, TRAJ_5}, {1296, 972, 10.5, Combine::Energy, Mode::Bsb, 1, TRAJ_6}, {1944, 972, 6, Combine::Energy, Mode::KSpin, 3, TRAJ_7}, {1944, 1296, 9, Combine::Energy, Mode::Bsb, 1, TRAJ_8}, {1944, 1458, 10.5, Combine::Energy, Mode::Bsb, 1, TRAJ_9}, {3888, 1944, 6, Combine::Energy, Mode::KSpin, 1, TRAJ_10}, {3888, 2592, 9, Combine::Energy, Mode::Bsb, 1, TRAJ_11}, {3888, 2916, 10.5, Combine::Energy, Mode::Bsb, 1, TRAJ_12}};

// Exact joint Gray-pair likelihood recovery. Both the Python reference and
// this exporter use the same bracket, bisection count and Newton steps.
static const double QAM_LEVELS[4] = {
    -3.0 / std::sqrt(10.0), -1.0 / std::sqrt(10.0),
     1.0 / std::sqrt(10.0),  3.0 / std::sqrt(10.0)
};
static const double QAM_BRACKET = 3.0 / std::sqrt(10.0) + 12.0;

static double log_sum_exp2(double first, double second) {
    const double peak = std::max(first, second);
    return peak + std::log1p(std::exp(-std::fabs(first - second)));
}

// Bisection needs millions of softplus values per frame. A cubic-interpolated
// table reproduces log1p(exp(-x)) to about 1e-10, far below the decoder's LLR
// resolution; the final consistency check still uses the exact log_sum_exp2.
static const int SOFTPLUS_STEPS = 1600;
static const double SOFTPLUS_MAX = 40.0;
static double softplus_table[SOFTPLUS_STEPS + 4];
static bool softplus_ready = false;

static void ensure_softplus_table() {
    if (softplus_ready) return;
    const double step = SOFTPLUS_MAX / SOFTPLUS_STEPS;
    for (int index = 0; index <= SOFTPLUS_STEPS + 3; ++index) {
        softplus_table[index] = std::log1p(std::exp(-step * index));
    }
    softplus_ready = true;
}

static double softplus_neg(double x) {
    if (x >= SOFTPLUS_MAX) return 0.0;
    ensure_softplus_table();
    const double scaled = x * SOFTPLUS_STEPS / SOFTPLUS_MAX;
    int index = static_cast<int>(scaled);
    if (index > SOFTPLUS_STEPS - 1) index = SOFTPLUS_STEPS - 1;
    const double t = scaled - index;
    const double p0 = softplus_table[index];
    const double p1 = softplus_table[index + 1];
    const double p2 = softplus_table[index + 2];
    const double p3 = softplus_table[index + 3];
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double l0 = (-t3 + 6.0 * t2 - 11.0 * t + 6.0) / 6.0;
    const double l1 = (t3 - 5.0 * t2 + 6.0 * t) / 2.0;
    const double l2 = (-t3 + 4.0 * t2 - 3.0 * t) / 2.0;
    const double l3 = (t3 - 3.0 * t2 + 2.0 * t) / 6.0;
    return p0 * l0 + p1 * l1 + p2 * l2 + p3 * l3;
}

static double fast_log_sum_exp2(double first, double second) {
    const double peak = std::max(first, second);
    return peak + softplus_neg(std::fabs(first - second));
}

// The common -y^2/(2 sigma^2) term cancels from every log-likelihood
// difference below, so precompute only the y-linear part of each component.
struct PairChannel {
    double slope[4];
    double intercept[4];
    double inv_sigma_sq;
};

static PairChannel make_pair_channel(double sigma) {
    PairChannel channel;
    channel.inv_sigma_sq = 1.0 / (sigma * sigma);
    for (int i = 0; i < 4; ++i) {
        const double level = QAM_LEVELS[i];
        channel.slope[i] = level * channel.inv_sigma_sq;
        channel.intercept[i] = -0.5 * level * level * channel.inv_sigma_sq;
    }
    return channel;
}

static void component_loglikes(double y, const PairChannel& channel, double out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = channel.slope[i] * y + channel.intercept[i];
    }
}

static double first_bit_llr_y(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return fast_log_sum_exp2(ell[0], ell[1]) - fast_log_sum_exp2(ell[2], ell[3]);
}

// K-spin bit identity requires the same exact log1p(exp(-x)) used by
// qam_pairs.py.  Non-K-spin cases keep the faster interpolated table.
static double first_bit_llr_y_exact(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return log_sum_exp2(ell[0], ell[1]) - log_sum_exp2(ell[2], ell[3]);
}

static double second_bit_llr_y(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return log_sum_exp2(ell[0], ell[3]) - log_sum_exp2(ell[1], ell[2]);
}

static double first_bit_llr_derivative(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    double peak = std::max(ell[0], ell[1]);
    double w0 = std::exp(ell[0] - peak);
    double w1 = std::exp(ell[1] - peak);
    const double mu_negative = (w0 * QAM_LEVELS[0] + w1 * QAM_LEVELS[1]) / (w0 + w1);
    peak = std::max(ell[2], ell[3]);
    w0 = std::exp(ell[2] - peak);
    w1 = std::exp(ell[3] - peak);
    const double mu_positive = (w0 * QAM_LEVELS[2] + w1 * QAM_LEVELS[3]) / (w0 + w1);
    return (mu_negative - mu_positive) * channel.inv_sigma_sq;
}

static double recover_amplitude(
    double target, const PairChannel& channel, bool exact = false
) {
    double low = -QAM_BRACKET;
    double high = QAM_BRACKET;
    for (int i = 0; i < 64; ++i) {
        const double middle = 0.5 * (low + high);
        const double value = exact
            ? first_bit_llr_y_exact(middle, channel)
            : first_bit_llr_y(middle, channel);
        if (value > target) {
            low = middle;
        } else {
            high = middle;
        }
    }
    double y = 0.5 * (low + high);
    for (int i = 0; i < 2; ++i) {
        const double derivative = first_bit_llr_derivative(y, channel);
        if (std::fabs(derivative) < 1e-300) break;
        const double value = exact
            ? first_bit_llr_y_exact(y, channel)
            : first_bit_llr_y(y, channel);
        const double step = (value - target) / derivative;
        y = std::min(QAM_BRACKET, std::max(-QAM_BRACKET, y - step));
    }
    return y;
}

struct PairTables {
    std::vector<double> linear;
    std::vector<double> d;
};

static PairTables build_pair_tables(
    const std::vector<double>& llr, double snr, double& bad_fraction,
    bool exact = false
) {
    const int n = static_cast<int>(llr.size());
    PairTables tables;
    tables.linear.assign(n, 0.0);
    tables.d.assign(n, 0.0);
    const double sigma = 1.0 / std::sqrt(2.0 * std::pow(10.0, snr / 10.0));
    const PairChannel channel = make_pair_channel(sigma);
    int bad = 0;
    for (int pair = 0; pair < n / 2; ++pair) {
        const int first = 2 * pair;
        const int second = first + 1;
        const double y = recover_amplitude(llr[first], channel, exact);
        double ell[4];
        component_loglikes(y, channel, ell);
        // Gray order: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
        const double a = (ell[0] + ell[1] - ell[3] - ell[2]) / 4.0;
        const double b = (ell[0] - ell[1] + ell[3] - ell[2]) / 4.0;
        const double d = (ell[0] - ell[1] - ell[3] + ell[2]) / 4.0;
        tables.linear[first] = 2.0 * a;
        tables.linear[second] = 2.0 * b;
        tables.d[first] = d;
        tables.d[second] = d;
        const double tolerance = 1e-6 * std::max(1.0, std::fabs(llr[second]));
        if (std::fabs(second_bit_llr_y(y, channel) - llr[second]) > tolerance) {
            ++bad;
        }
    }
    bad_fraction = (n > 0) ? static_cast<double>(bad) / (n / 2) : 0.0;
    return tables;
}

static std::vector<int> run_trajectory(
    const std::vector<double>& llr,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    const TrajectoryConfig& cfg,
    std::vector<double>& x_out,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d
) {
    const int n = static_cast<int>(llr.size());
    const bool joint = (cfg.joint_pairs == 1 && pair_linear != nullptr && pair_d != nullptr);
    const std::vector<double>& channel = joint ? *pair_linear : llr;
    const int m = static_cast<int>(starts.size()) - 1;
    std::vector<double> spins(n);
    std::vector<double> weighted(n, 0.0);
    std::vector<double> force(n);
    std::vector<double> x(n);
    std::vector<double> y(n);
    int max_deg = 0;
    for (int row = 0; row < m; ++row) {
        max_deg = std::max(max_deg, starts[row + 1] - starts[row]);
    }
    std::vector<double> pref(max_deg + 1, 1.0);
    std::vector<double> suff(max_deg + 1, 1.0);
    const bool use_reaction = (cfg.beta != 0.0 || cfg.beta_values != nullptr);
    const bool use_scale = (cfg.parity_scale != 1.0);
    const int parity_start = n - m;  // H=[A|I]
    std::vector<double> reaction(use_reaction ? n : 0, 0.0);
    std::vector<double> qq(max_deg + 1), uu(max_deg + 1);
    std::vector<double> pp(max_deg + 1), qp(max_deg + 1), ps(max_deg + 1), qs(max_deg + 1);

    const double tau = cfg.tau_values ? cfg.tau_values[0] : cfg.tau;
    for (int col = 0; col < n; ++col) {
        x[col] = std::tanh(llr[col] / tau);
        y[col] = 0.0;
    }

    for (int step = 0; step < STEPS; ++step) {
        const double xi = cfg.xi_values ? cfg.xi_values[step] : cfg.xi;
        const double lambd = cfg.lambd_values ? cfg.lambd_values[step] : cfg.lambd;
        const double delta = cfg.delta_values ? cfg.delta_values[step] : cfg.delta;
        const double dt = cfg.dt_values ? cfg.dt_values[step] : cfg.dt;
        const double beta = cfg.beta_values ? cfg.beta_values[step] : cfg.beta;
        for (int col = 0; col < n; ++col) {
            if (cfg.mode == Mode::Dsb) {
                spins[col] = (x[col] >= 0.0) ? 1.0 : -1.0;
            } else {
                double value = std::min(1.0, std::max(-1.0, x[col]));
                if (std::fabs(value) < EPS) {
                    value = (value >= 0.0) ? EPS : -EPS;
                }
                spins[col] = value;
            }
        }
        std::fill(weighted.begin(), weighted.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            const int start = starts[row];
            const int deg = starts[row + 1] - start;
            if (deg <= 0) continue;
            pref[0] = 1.0;
            for (int i = 0; i < deg; ++i) {
                pref[i + 1] = pref[i] * spins[cols[start + i]];
            }
            suff[deg] = 1.0;
            for (int i = deg - 1; i >= 0; --i) {
                suff[i] = suff[i + 1] * spins[cols[start + i]];
            }
            for (int i = 0; i < deg; ++i) {
                weighted[cols[start + i]] += pref[i] * suff[i + 1];
            }
        }
        if (use_reaction) {
            // sum_c sum_{j in c\i} (1-x_j^2) prod_{l in c\{i,j}} x_l^2 ; gradient of a term constant on +-1 spins
            std::fill(reaction.begin(), reaction.end(), 0.0);
            for (int row = 0; row < m; ++row) {
                const int start = starts[row];
                const int deg = starts[row + 1] - start;
                if (deg <= 0) continue;
                for (int t = 0; t < deg; ++t) {
                    const double s = spins[cols[start + t]];
                    qq[t] = s * s;
                    uu[t] = 1.0 - qq[t];
                }
                pp[0] = 1.0; qp[0] = 0.0;
                for (int t = 0; t < deg; ++t) {
                    pp[t + 1] = pp[t] * qq[t];
                    qp[t + 1] = qp[t] * qq[t] + pp[t] * uu[t];
                }
                ps[deg] = 1.0; qs[deg] = 0.0;
                for (int t = deg - 1; t >= 0; --t) {
                    ps[t] = ps[t + 1] * qq[t];
                    qs[t] = qs[t + 1] * qq[t] + ps[t + 1] * uu[t];
                }
                for (int t = 0; t < deg; ++t) {
                    reaction[cols[start + t]] += qp[t] * ps[t + 1] + pp[t] * qs[t + 1];
                }
            }
        }
        const double schedule = static_cast<double>(step) / (STEPS - 1);
        const double coupling = 0.5 * lambd;
        for (int col = 0; col < n; ++col) {
            const double scale = (use_scale && col >= parity_start) ? cfg.parity_scale : 1.0;
            if (use_scale) {
                force[col] = channel[col] + (coupling * scale) * weighted[col];
            } else {
                force[col] = channel[col] + coupling * weighted[col];
            }
            if (joint) {
                const int partner = col ^ 1;
                const double other = spins[partner];
                const double strength = (*pair_d)[col];
                force[col] += 2.0 * cfg.pair_scale * strength * other;
                if (cfg.pair_gamma != 0.0) {
                    force[col] -= 4.0 * cfg.pair_gamma * strength * strength
                                  * (1.0 - other * other) * spins[col];
                }
            }
            if (beta != 0.0) {
                double r = ((beta * (coupling * coupling)) * spins[col]) * reaction[col];
                if (use_scale) r = r * scale;
                force[col] = force[col] - r;
            }
            const double pump = -(delta - schedule) * x[col];
            const double y_new = y[col] + dt * (pump + xi * force[col]);
            const double x_new = x[col] + dt * y_new;
            if (std::fabs(x_new) > 1.0) {
                x[col] = (x_new >= 0.0) ? 1.0 : -1.0;
                y[col] = 0.0;
            } else {
                x[col] = x_new;
                y[col] = y_new;
            }
        }
    }

    std::vector<int> hard(n);
    for (int col = 0; col < n; ++col) {
        hard[col] = (x[col] < 0.0) ? 1 : 0;
    }
    x_out = x;
    return hard;
}

static std::vector<double> kspin_parity_weights(
    const std::vector<double>& llr, int k, int m, double g, double clip
) {
    std::vector<double> weights(static_cast<size_t>(m));
    for (int row = 0; row < m; ++row) {
        weights[static_cast<size_t>(row)] =
            g * clip * std::tanh(llr[static_cast<size_t>(k + row)] / clip);
    }
    return weights;
}

static std::vector<int> run_kspin_trajectory(
    const std::vector<double>& llr,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols,
    const TrajectoryConfig& cfg,
    std::vector<double>& x_out,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d
) {
    const int n = static_cast<int>(llr.size());
    const int m = n - k;
    const bool joint = (cfg.joint_pairs == 1 && pair_linear != nullptr && pair_d != nullptr);
    std::vector<double> x(static_cast<size_t>(k));
    std::vector<double> y(static_cast<size_t>(k), 0.0);
    std::vector<double> spins(static_cast<size_t>(k));
    std::vector<double> weighted(static_cast<size_t>(k), 0.0);
    std::vector<double> reaction(static_cast<size_t>(k), 0.0);
    std::vector<double> field(static_cast<size_t>(k), 0.0);
    std::vector<double> force(static_cast<size_t>(k), 0.0);
    std::vector<double> weights = kspin_parity_weights(llr, k, m, cfg.g, cfg.clip);

    int max_deg = 0;
    for (int row = 0; row < m; ++row) {
        max_deg = std::max(max_deg, info_starts[static_cast<size_t>(row + 1)]
                                    - info_starts[static_cast<size_t>(row)]);
    }
    std::vector<double> pref(static_cast<size_t>(max_deg + 1), 1.0);
    std::vector<double> suff(static_cast<size_t>(max_deg + 1), 1.0);
    std::vector<double> qq(static_cast<size_t>(max_deg + 1));
    std::vector<double> uu(static_cast<size_t>(max_deg + 1));
    std::vector<double> pp(static_cast<size_t>(max_deg + 1));
    std::vector<double> qp(static_cast<size_t>(max_deg + 1));
    std::vector<double> ps(static_cast<size_t>(max_deg + 1));
    std::vector<double> qs(static_cast<size_t>(max_deg + 1));

    const double tau = cfg.tau_values ? cfg.tau_values[0] : cfg.tau;
    for (int col = 0; col < k; ++col) {
        x[static_cast<size_t>(col)] = std::tanh(llr[static_cast<size_t>(col)] / tau);
    }

    for (int step = 0; step < STEPS; ++step) {
        for (int col = 0; col < k; ++col) {
            double value = std::min(1.0, std::max(-1.0, x[static_cast<size_t>(col)]));
            if (std::fabs(value) < EPS) {
                value = (value >= 0.0) ? EPS : -EPS;
            }
            spins[static_cast<size_t>(col)] = value;
        }
        for (int col = 0; col < k; ++col) {
            if (joint) {
                const int partner = col ^ 1;
                const double value = spins[static_cast<size_t>(col)];
                const double other = spins[static_cast<size_t>(partner)];
                const double d = (*pair_d)[static_cast<size_t>(col)];
                double value_field = (*pair_linear)[static_cast<size_t>(col)] + 2.0 * d * other;
                if (cfg.pair_gamma != 0.0) {
                    value_field -= 4.0 * cfg.pair_gamma * d * d
                                   * (1.0 - other * other) * value;
                }
                field[static_cast<size_t>(col)] = value_field;
            } else {
                field[static_cast<size_t>(col)] = llr[static_cast<size_t>(col)];
            }
        }

        std::fill(weighted.begin(), weighted.end(), 0.0);
        std::fill(reaction.begin(), reaction.end(), 0.0);
        const bool use_reaction = (cfg.beta != 0.0);
        for (int row = 0; row < m; ++row) {
            const int start = info_starts[static_cast<size_t>(row)];
            const int deg = info_starts[static_cast<size_t>(row + 1)] - start;
            if (deg <= 0) continue;
            pref[0] = 1.0;
            for (int index = 0; index < deg; ++index) {
                pref[static_cast<size_t>(index + 1)] =
                    pref[static_cast<size_t>(index)]
                    * spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
            }
            suff[static_cast<size_t>(deg)] = 1.0;
            for (int index = deg - 1; index >= 0; --index) {
                suff[static_cast<size_t>(index)] =
                    suff[static_cast<size_t>(index + 1)]
                    * spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
            }
            for (int index = 0; index < deg; ++index) {
                const int col = info_cols[static_cast<size_t>(start + index)];
                weighted[static_cast<size_t>(col)] +=
                    weights[static_cast<size_t>(row)] * pref[static_cast<size_t>(index)]
                    * suff[static_cast<size_t>(index + 1)];
            }
            if (!use_reaction) continue;
            for (int index = 0; index < deg; ++index) {
                const double value =
                    spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
                qq[static_cast<size_t>(index)] = value * value;
                uu[static_cast<size_t>(index)] = 1.0 - qq[static_cast<size_t>(index)];
            }
            pp[0] = 1.0; qp[0] = 0.0;
            for (int index = 0; index < deg; ++index) {
                pp[static_cast<size_t>(index + 1)] =
                    pp[static_cast<size_t>(index)] * qq[static_cast<size_t>(index)];
                qp[static_cast<size_t>(index + 1)] =
                    qp[static_cast<size_t>(index)] * qq[static_cast<size_t>(index)]
                    + pp[static_cast<size_t>(index)] * uu[static_cast<size_t>(index)];
            }
            ps[static_cast<size_t>(deg)] = 1.0; qs[static_cast<size_t>(deg)] = 0.0;
            for (int index = deg - 1; index >= 0; --index) {
                ps[static_cast<size_t>(index)] =
                    ps[static_cast<size_t>(index + 1)] * qq[static_cast<size_t>(index)];
                qs[static_cast<size_t>(index)] =
                    qs[static_cast<size_t>(index + 1)] * qq[static_cast<size_t>(index)]
                    + ps[static_cast<size_t>(index + 1)] * uu[static_cast<size_t>(index)];
            }
            const double w = weights[static_cast<size_t>(row)];
            for (int index = 0; index < deg; ++index) {
                const int col = info_cols[static_cast<size_t>(start + index)];
                const double value = spins[static_cast<size_t>(col)];
                reaction[static_cast<size_t>(col)] += cfg.beta * w * w * value
                    * (qp[static_cast<size_t>(index)] * ps[static_cast<size_t>(index + 1)]
                       + pp[static_cast<size_t>(index)] * qs[static_cast<size_t>(index + 1)]);
            }
        }

        const double xi = cfg.xi_values ? cfg.xi_values[step] : cfg.xi;
        const double delta = cfg.delta_values ? cfg.delta_values[step] : cfg.delta;
        const double dt = cfg.dt_values ? cfg.dt_values[step] : cfg.dt;
        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < k; ++col) {
            const double f = field[static_cast<size_t>(col)]
                             + weighted[static_cast<size_t>(col)]
                             - reaction[static_cast<size_t>(col)];
            const double pump = -(delta - schedule) * x[static_cast<size_t>(col)];
            const double y_new = y[static_cast<size_t>(col)] + dt * (pump + xi * f);
            const double x_new = x[static_cast<size_t>(col)] + dt * y_new;
            if (std::fabs(x_new) > 1.0) {
                x[static_cast<size_t>(col)] = (x_new >= 0.0) ? 1.0 : -1.0;
                y[static_cast<size_t>(col)] = 0.0;
            } else {
                x[static_cast<size_t>(col)] = x_new;
                y[static_cast<size_t>(col)] = y_new;
            }
        }
    }

    std::vector<int> hard(static_cast<size_t>(k));
    for (int col = 0; col < k; ++col) {
        hard[static_cast<size_t>(col)] = (x[static_cast<size_t>(col)] < 0.0) ? 1 : 0;
    }
    x_out = x;
    return hard;
}

static double kspin_energy_score(
    const std::vector<int>& info_hard,
    const std::vector<double>& llr,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols,
    const std::vector<double>& clipped_weights,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d,
    int select_unclipped
) {
    const int m = static_cast<int>(clipped_weights.size());
    std::vector<double> spins(static_cast<size_t>(k));
    for (int col = 0; col < k; ++col) {
        spins[static_cast<size_t>(col)] =
            (info_hard[static_cast<size_t>(col)] == 0) ? 1.0 : -1.0;
    }
    double value = 0.0;
    for (int col = 0; col < k; ++col) {
        double field = llr[static_cast<size_t>(col)];
        if (pair_linear != nullptr && pair_d != nullptr) {
            field = (*pair_linear)[static_cast<size_t>(col)]
                    + 2.0 * (*pair_d)[static_cast<size_t>(col)]
                    * spins[static_cast<size_t>(col ^ 1)];
        }
        value -= field * spins[static_cast<size_t>(col)];
    }
    for (int row = 0; row < m; ++row) {
        double product = 1.0;
        for (int pos = info_starts[static_cast<size_t>(row)];
             pos < info_starts[static_cast<size_t>(row + 1)]; ++pos) {
            product *= spins[static_cast<size_t>(info_cols[static_cast<size_t>(pos)])];
        }
        const double weight = select_unclipped
            ? llr[static_cast<size_t>(k + row)]
            : clipped_weights[static_cast<size_t>(row)];
        value -= weight * product;
    }
    return value;
}

static std::vector<int> encode_kspin_parity(
    const std::vector<int>& info_hard,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols
) {
    const int m = static_cast<int>(info_starts.size()) - 1;
    const int n = k + m;
    std::vector<int> hard(static_cast<size_t>(n));
    for (int col = 0; col < k; ++col) {
        hard[static_cast<size_t>(col)] = info_hard[static_cast<size_t>(col)];
    }
    for (int row = 0; row < m; ++row) {
        int parity = 0;
        for (int pos = info_starts[static_cast<size_t>(row)];
             pos < info_starts[static_cast<size_t>(row + 1)]; ++pos) {
            parity ^= info_hard[static_cast<size_t>(info_cols[static_cast<size_t>(pos)])];
        }
        hard[static_cast<size_t>(k + row)] = parity;
    }
    return hard;
}

static double energy_score(
    const std::vector<int>& hard,
    const std::vector<double>& channel,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    double lambd,
    const std::vector<double>* pair_d,
    double pair_scale
) {
    const int n = static_cast<int>(channel.size());
    const int m = static_cast<int>(starts.size()) - 1;
    std::vector<double> spins(n);
    for (int col = 0; col < n; ++col) {
        spins[col] = (hard[col] == 0) ? 1.0 : -1.0;
    }
    double channel_energy = 0.0;
    for (int col = 0; col < n; ++col) {
        channel_energy -= channel[col] * spins[col];
    }
    if (pair_d != nullptr) {
        for (int pair = 0; pair < n / 2; ++pair) {
            channel_energy -= (2.0 * pair_scale * (*pair_d)[2 * pair])
                              * spins[2 * pair] * spins[2 * pair + 1];
        }
    }
    double penalty = 0.0;
    for (int row = 0; row < m; ++row) {
        double product = 1.0;
        for (int pos = starts[row]; pos < starts[row + 1]; ++pos) {
            product *= spins[cols[pos]];
        }
        penalty += 1.0 - product;
    }
    return channel_energy + 0.5 * lambd * penalty;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n = 0;
    int k = 0;
    int b = 0;
    double snr = 0.0;
    if (!(std::cin >> n >> k >> b >> snr)) {
        return 0;
    }
    const int m = n - k;
    if (m < 0) {
        return 0;
    }

    std::vector<int> starts(static_cast<size_t>(m) + 1, 0);
    std::vector<int> cols;
    cols.reserve(static_cast<size_t>(m) * 8);
    std::vector<int> info_starts(static_cast<size_t>(m) + 1, 0);
    std::vector<int> info_cols;
    info_cols.reserve(static_cast<size_t>(m) * 8);
    for (int row = 0; row < m; ++row) {
        starts[row] = static_cast<int>(cols.size());
        info_starts[row] = static_cast<int>(info_cols.size());
        for (int col = 0; col < n; ++col) {
            int value = 0;
            std::cin >> value;
            if (value != 0) {
                cols.push_back(col);
                if (col < k) info_cols.push_back(col);
            }
        }
    }
    starts[m] = static_cast<int>(cols.size());
    info_starts[m] = static_cast<int>(info_cols.size());

    std::vector<std::vector<double>> llr(
        static_cast<size_t>(b), std::vector<double>(static_cast<size_t>(n))
    );
    for (int sample = 0; sample < b; ++sample) {
        for (int col = 0; col < n; ++col) {
            std::cin >> llr[static_cast<size_t>(sample)][col];
        }
    }
    int unused = 0;
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cin >> unused;
        }
    }
    static_cast<void>(unused);

    const CaseConfig* active = &FALLBACK;
    double best_diff = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < static_cast<size_t>(12); ++index) {
        const CaseConfig& candidate = CASES[index];
        if (candidate.n != n || candidate.k != k) {
            continue;
        }
        const double diff = std::fabs(candidate.snr - snr);
        if (diff < best_diff) {
            best_diff = diff;
            active = &candidate;
        }
    }

    const bool case_kspin = (active->mode == Mode::KSpin);
    bool case_joint = false;
    for (int index = 0; index < active->count; ++index) {
        if (active->trajectories[index].joint_pairs == 1) case_joint = true;
    }

    for (int sample = 0; sample < b; ++sample) {
        const std::vector<double>& row = llr[static_cast<size_t>(sample)];
        const std::vector<double>* row_channel = &row;
        std::vector<PairTables> frame_tables;
        bool frame_joint = case_joint;
        if (case_joint) {
            double bad_fraction = 0.0;
            frame_tables.push_back(
                build_pair_tables(row, active->snr, bad_fraction, case_kspin)
            );
            if (bad_fraction > 0.001) {
                frame_joint = false;
            } else {
                row_channel = &frame_tables[0].linear;
            }
        }
        const std::vector<double>* frame_d =
            frame_joint ? &frame_tables[0].d : nullptr;
        std::vector<int> decided;

        if (case_kspin) {
            const int k = active->k;
            std::vector<std::vector<int>> info_hards(
                static_cast<size_t>(active->count), std::vector<int>(static_cast<size_t>(k))
            );
            std::vector<std::vector<double>> softs(
                static_cast<size_t>(active->count), std::vector<double>(static_cast<size_t>(k))
            );
            std::vector<std::vector<double>> kweights(
                static_cast<size_t>(active->count), std::vector<double>(static_cast<size_t>(m))
            );
            if (active->count > 1) {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(active->count));
                for (int index = 0; index < active->count; ++index) {
                    workers.emplace_back([&, index]() {
                        const TrajectoryConfig& cfg = active->trajectories[static_cast<size_t>(index)];
                        kweights[static_cast<size_t>(index)] = kspin_parity_weights(
                            row, k, m, cfg.g, cfg.clip
                        );
                        const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                        info_hards[static_cast<size_t>(index)] = run_kspin_trajectory(
                            row, k, info_starts, info_cols, cfg,
                            softs[static_cast<size_t>(index)],
                            traj_joint ? &frame_tables[0].linear : nullptr,
                            traj_joint ? &frame_tables[0].d : nullptr
                        );
                    });
                }
                for (std::thread& worker : workers) {
                    worker.join();
                }
            } else {
                const TrajectoryConfig& cfg = active->trajectories[0];
                kweights[0] = kspin_parity_weights(row, k, m, cfg.g, cfg.clip);
                const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                info_hards[0] = run_kspin_trajectory(
                    row, k, info_starts, info_cols, cfg, softs[0],
                    traj_joint ? &frame_tables[0].linear : nullptr,
                    traj_joint ? &frame_tables[0].d : nullptr
                );
            }

            std::vector<int> selected_info = info_hards[0];
            if (active->combine == Combine::Energy) {
                double best_energy = kspin_energy_score(
                    selected_info, row, k, info_starts, info_cols, kweights[0],
                    active->trajectories[0].joint_pairs == 1 && frame_joint
                        ? &frame_tables[0].linear : nullptr,
                    active->trajectories[0].joint_pairs == 1 && frame_joint
                        ? &frame_tables[0].d : nullptr,
                    active->trajectories[0].select_unclipped
                );
                for (int index = 1; index < active->count; ++index) {
                    const TrajectoryConfig& cfg = active->trajectories[static_cast<size_t>(index)];
                    const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                    const double value = kspin_energy_score(
                        info_hards[static_cast<size_t>(index)], row, k,
                        info_starts, info_cols, kweights[static_cast<size_t>(index)],
                        traj_joint ? &frame_tables[0].linear : nullptr,
                        traj_joint ? &frame_tables[0].d : nullptr,
                        cfg.select_unclipped
                    );
                    if (value < best_energy) {
                        best_energy = value;
                        selected_info = info_hards[static_cast<size_t>(index)];
                    }
                }
            } else if (active->combine == Combine::Vote) {
                for (int col = 0; col < k; ++col) {
                    int ones = 0;
                    for (int index = 0; index < active->count; ++index) {
                        ones += info_hards[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    selected_info[static_cast<size_t>(col)] = (2 * ones >= active->count) ? 1 : 0;
                }
            } else {
                for (int col = 0; col < k; ++col) {
                    double total = 0.0;
                    for (int index = 0; index < active->count; ++index) {
                        total += softs[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    selected_info[static_cast<size_t>(col)] = (total < 0.0) ? 1 : 0;
                }
            }
            decided = encode_kspin_parity(selected_info, k, info_starts, info_cols);
        } else {
            std::vector<std::vector<int>> hards(static_cast<size_t>(active->count));
            std::vector<std::vector<double>> softs(static_cast<size_t>(active->count));
            std::vector<double> lambdas(static_cast<size_t>(active->count));
            if (active->count > 1) {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(active->count));
                for (int index = 0; index < active->count; ++index) {
                    workers.emplace_back([&, index]() {
                        hards[static_cast<size_t>(index)] = run_trajectory(
                            row, starts, cols, active->trajectories[index],
                            softs[static_cast<size_t>(index)], row_channel, frame_d
                        );
                        lambdas[static_cast<size_t>(index)] =
                            active->trajectories[index].energy_lambda;
                    });
                }
                for (std::thread& worker : workers) {
                    worker.join();
                }
            } else {
                hards[0] = run_trajectory(
                    row, starts, cols, active->trajectories[0], softs[0],
                    row_channel, frame_d
                );
                lambdas[0] = active->trajectories[0].energy_lambda;
            }

            decided = hards[0];
            if (active->combine == Combine::Energy) {
                double best_energy = energy_score(
                    decided, *row_channel, starts, cols, lambdas[0],
                    active->trajectories[0].joint_pairs == 1 ? frame_d : nullptr,
                    active->trajectories[0].pair_scale
                );
                for (int index = 1; index < active->count; ++index) {
                    const double value = energy_score(
                        hards[static_cast<size_t>(index)], *row_channel, starts, cols,
                        lambdas[static_cast<size_t>(index)],
                        active->trajectories[static_cast<size_t>(index)].joint_pairs == 1
                            ? frame_d : nullptr,
                        active->trajectories[static_cast<size_t>(index)].pair_scale
                    );
                    if (value < best_energy) {
                        best_energy = value;
                        decided = hards[static_cast<size_t>(index)];
                    }
                }
            } else if (active->combine == Combine::Vote) {
                for (int col = 0; col < n; ++col) {
                    int ones = 0;
                    for (int index = 0; index < active->count; ++index) {
                        ones += hards[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    decided[static_cast<size_t>(col)] = (2 * ones >= active->count) ? 1 : 0;
                }
            } else {
                for (int col = 0; col < n; ++col) {
                    double total = 0.0;
                    for (int index = 0; index < active->count; ++index) {
                        total += softs[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    decided[static_cast<size_t>(col)] = (total < 0.0) ? 1 : 0;
                }
            }
        }

        for (int col = 0; col < n; ++col) {
            std::cout << decided[static_cast<size_t>(col)];
            std::cout << ((col + 1 == n) ? '\n' : ' ');
        }
    }
    return 0;
}
