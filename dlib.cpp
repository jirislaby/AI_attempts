#if MATRIX
#include <dlib/matrix.h>
#include <iostream>
#include <vector>

using namespace dlib;

// Funkce pro aktualizaci latentních faktorů
void update_factors(matrix<double>& X, const matrix<double>& Y, const matrix<double>& ratings, double lambda) {
    for (long i = 0; i < X.nr(); ++i) {
        matrix<double> Ai = trans(Y) * Y + lambda * identity_matrix<double>(Y.nc());
        matrix<double> Vi = trans(Y) * colm(ratings, i);  // Vyber sloupec i
        set_rowm(X, i) = trans(inv(Ai) * Vi);
    }
}

// Alternating Least Squares (ALS)
void als(matrix<double>& user_factors, matrix<double>& item_factors, const matrix<double>& ratings, double lambda, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Aktualizace latentních faktorů pro uživatele
        update_factors(user_factors, item_factors, ratings, lambda);
        // Aktualizace latentních faktorů pro položky
        update_factors(item_factors, user_factors, trans(ratings), lambda);
    }
}

int main() {
    // Vytvoření a inicializace matice hodnocení
    matrix<double> ratings;
    ratings.set_size(4, 3);  // Matice o velikosti 4x3 (4 vývojáři, 3 soubory)

    // Vyplnění matice hodnocení (řádky jsou vývojáři, sloupce jsou soubory, hodnoty jsou počty commitů)
    ratings = 5, 0, 0,
              4, 0, 0,
              0, 0, 3,
              0, 5, 0;

    // Parametry pro ALS
    int num_users = ratings.nr();   // Počet uživatelů (řádky)
    int num_items = ratings.nc();   // Počet položek (sloupce)
    int num_factors = 2;            // Počet latentních faktorů
    double lambda = 0.1;            // Regularizace
    int iterations = 10;            // Počet iterací

    // Inicializace latentních faktorů pro uživatele a položky
    matrix<double> user_factors = randm(num_users, num_factors);
    matrix<double> item_factors = randm(num_items, num_factors);

    // Spuštění ALS
    als(user_factors, item_factors, ratings, lambda, iterations);

    // Výpis latentních faktorů po trénování
    std::cout << "Latentní faktory uživatelů (vývojáři):\n" << user_factors << std::endl;
    std::cout << "Latentní faktory položek (soubory):\n" << item_factors << std::endl;

    // Predikce: vývojář 0 a soubor 1
    matrix<double> predictions = user_factors * trans(item_factors);
    std::cout << "Predikovaná hodnota (vývojář 0, soubor 1): " << predictions(0, 1) << std::endl;

    return 0;
}
#else
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/rand.h>
#include <iostream>
#include <vector>

using namespace dlib;

typedef matrix<double,2,1> sample_type;
typedef long label_type;

// Funkce pro generování tréninkových dat
void generate_data(std::vector<sample_type>& samples, std::vector<label_type>& labels) {
    // Příklad generování dat: 2 třídy (0 a 1)
    samples.push_back({1.0, 2.0});
    labels.push_back(0);
    samples.push_back({1.5, 2.5});
    labels.push_back(0);
    samples.push_back({3.0, 3.0});
    labels.push_back(1);
    samples.push_back({3.5, 3.5});
    labels.push_back(1);
}

int main() {
    // Inicializace vzorků a štítků
    std::vector<sample_type> samples;
    std::vector<label_type> labels;

    // Generování tréninkových dat
    generate_data(samples, labels);
    matrix<double> test_sample = {2.0, 2.0}; // Testovací vzorek
#if 0
    // Trénink modelu pro multiklasifikaci
    svm_multiclass_linear_trainer<linear_kernel<matrix<double>>> trainer;
    decision_function<linear_kernel<matrix<double>>> df = trainer.train(samples, labels);

    // Testování modelu
    /*
    long predicted_label = df(test_sample);
    std::cout << "Predikovaná třída: " << predicted_label << std::endl;
*/
#endif
    // Příklad použití One-vs-One trénování
    typedef radial_basis_kernel<sample_type> rbf_kernel;
    krr_trainer<rbf_kernel> rbf_trainer;

    one_vs_one_trainer<any_trainer<sample_type>, label_type> trainer;
    trainer.set_trainer(rbf_trainer);
    auto ovo_df = trainer.train(samples, labels);
    // Testování One-vs-One modelu
    long ovo_predicted_label = ovo_df(test_sample);
    std::cout << "Predikovaná třída (One-vs-One): " << ovo_predicted_label << std::endl;

    return 0;
}
#endif
