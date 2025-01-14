#include "embed.h"

unsigned Embedding::findOrCreateTokenId(const std::string &token)
{
	unsigned tokenId;
	auto tokIt = tokenToId.find(token);
	if (tokIt == tokenToId.end()) {
		tokenId = tokenToId.size();
		tokenToId[token] = tokenId;
		return tokenId;
	}

	return tokIt->second;
}

void Embedding::tokenizePath(const std::string &path, std::vector<unsigned> &tokens)
{
	size_t pos = 0, found;
	while ((found = path.find_first_of('/', pos)) != std::string::npos) {
		tokens.push_back(findOrCreateTokenId(path.substr(pos, found - pos)));
		pos = found + 1;
	}
	tokens.push_back(findOrCreateTokenId(path.substr(pos)));
}


dlib::matrix<double, 0, 1> Embedding::getEmbedding(const std::vector<unsigned> &path)
{
	dlib::matrix<double, 0, 1> one(tokenToId.size());

	one = 0;

	for (auto &tokenId: path)
		one(tokenId) = 1;

	return one;
}

dlib::matrix<double, 0, 1> Embedding::getEmbedding(const std::string &path)
{
	std::vector<unsigned> pathTokenized;

	tokenizePath(path, pathTokenized);

	return getEmbedding(pathTokenized);
}

void Embedding::getEmbedding(const std::vector<std::string> &paths,
			     std::vector<dlib::matrix<double, 0, 1>> &embedding)
{
	std::vector<std::vector<unsigned>> pathsTokenized;

	for (auto &path: paths) {
		std::vector<unsigned> pathTokenized;
		tokenizePath(path, pathTokenized);
		pathsTokenized.push_back(std::move(pathTokenized));
	}

	for (auto &path: pathsTokenized)
		embedding.push_back(std::move(getEmbedding(path)));
}
