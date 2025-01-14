#ifndef EMBED_H
#define EMBED_H

#include <unordered_map>
#include <vector>
#include <string>

#include <dlib/matrix.h>

class Embedding {
public:
	void getEmbedding(const std::vector<std::string> &paths,
			  std::vector<dlib::matrix<double, 0, 1>> &embedding);
	dlib::matrix<double, 0, 1> getEmbedding(const std::string &path);
private:
	unsigned findOrCreateTokenId(const std::string &str);
	void tokenizePath(const std::string &path, std::vector<unsigned> &tokens);
	dlib::matrix<double, 0, 1> getEmbedding(const std::vector<unsigned int> &path);

	std::unordered_map<std::string, unsigned> tokenToId;
};

#endif
