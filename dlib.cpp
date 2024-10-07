#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/rand.h>
#include <iostream>
#include <map>
#include <vector>

#include "sqlconn.h"

using namespace dlib;

typedef matrix<double,1,1> sample_type;
typedef double label_type;

int main()
{
	SQLConn sqlConn;
	std::vector<sample_type> samples;
	std::vector<label_type> labels;
	//std::map<unsigned, std::string> fileMap;
	std::map<unsigned, std::string> userMap;
	std::map<std::string, unsigned> pathMap;

	if (sqlConn.open("/tmp/conf_file_map.sqlite") < 0)
		return EXIT_FAILURE;

	int ret;
	auto sel = sqlConn.getSelMap();
	while ((ret = sqlite3_step(sel)) == SQLITE_ROW) {
		auto userId = sqlite3_column_int(sel, 0);
		auto fileId = sqlite3_column_int(sel, 1);
		//auto count = sqlite3_column_int(sel, 2);
		std::string email((const char *)sqlite3_column_text(sel, 3));
		std::string path((const char *)sqlite3_column_text(sel, 4));
		/*std::cout << "user=" << sqlite3_column_int(sel, 0) <<
			     " file=" << sqlite3_column_int(sel, 1) <<
			     " count=" << sqlite3_column_int(sel, 2) << "\n";*/
		samples.push_back({ (double)fileId*1000 });//, (double)count });
		labels.push_back(userId);
		//fileMap[fileId] = path;
		pathMap[path] = fileId;
		userMap[userId] = email;
	}

	if (ret != SQLITE_DONE) {
		std::cerr << "db step sel failed: " << sqlite3_errstr(ret) <<
			     " -> " << sqlite3_errmsg(sqlConn.getSql()) << "\n";
		return EXIT_FAILURE;
	}

	sample_type test_sample;
	test_sample(0, 0) = pathMap[std::string("fs/cifs/file.c")];

	const long num_training_samples = samples.size() * 0.99;

	std::vector<sample_type> training_samples(samples.begin(), samples.begin() + num_training_samples);
	std::vector<label_type> training_labels(labels.begin(), labels.begin() + num_training_samples);
	std::vector<sample_type> testing_samples(samples.begin() + num_training_samples, samples.end());
	std::vector<label_type> testing_labels(labels.begin() + num_training_samples, labels.end());
#if 1
	one_vs_one_trainer<any_trainer<sample_type>, label_type> trainer;
#if 0
	typedef radial_basis_kernel<sample_type> rbf_kernel;
	krr_trainer<rbf_kernel> rbf_trainer;
	trainer.set_trainer(rbf_trainer);
#else
	typedef linear_kernel<sample_type> kernel_type;
	svm_c_trainer<kernel_type> linear_trainer;
	trainer.set_trainer(linear_trainer);
#endif

	auto df = trainer.train(samples, labels);
	long ovo_predicted_label = df(test_sample);
	std::cout << "Predikovaná třída (One-vs-One): " << ovo_predicted_label << std::endl;
#else
	svm_c_trainer<linear_kernel<sample_type>> trainer;
	//trainer.set_c(10);
	std::cout << "XXX\n";
	auto df = trainer.train(training_samples, training_labels);
	auto out = df(test_sample);
	std::cout << "Predikovaná třída (One-vs-One): " << out << std::endl;
	std::cout << "\t" << userMap[(unsigned)out] << std::endl;
#endif

	int num_correct = 0;
	for (size_t i = 0; i < testing_samples.size(); ++i) {
	    label_type predicted_label = df(testing_samples[i]);// > 0 ? 1 : 0;
	    std::cout << "expected: " << testing_labels[i] << " got: " << predicted_label << "\n";
	    if (predicted_label == testing_labels[i])
		++num_correct;
	}

	std::cout << "Accuracy: " << num_correct << " <- " << num_correct / (double)testing_samples.size() << std::endl;

	return 0;
}
