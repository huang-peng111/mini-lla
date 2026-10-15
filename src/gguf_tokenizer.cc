#include "mini_llama/gguf_tokenizer.h"
#include <fstream>
namespace mini_lla {
bool VocabularyTokenizer::load(const std::string& path) {
  std::ifstream file(path); if (!file) return false;
  std::string token; int id; while (file >> id >> token) { token_to_id_[token] = id; id_to_token_[id] = token; }
  return !token_to_id_.empty();
}
std::vector<int> VocabularyTokenizer::encode(const std::string& text) const {
  std::vector<int> result;
  for (size_t offset = 0; offset < text.size();) { size_t best = 0; int best_id = -1; for (const auto& item : token_to_id_) if (item.first.size() > best && text.compare(offset, item.first.size(), item.first) == 0) { best = item.first.size(); best_id = item.second; } if (best == 0) { ++offset; } else { result.push_back(best_id); offset += best; } }
  return result;
}
std::string VocabularyTokenizer::decode(const std::vector<int>& tokens) const { std::string result; for (int token : tokens) { auto it = id_to_token_.find(token); if (it != id_to_token_.end()) result += it->second; } return result; }
}  // namespace mini_lla
