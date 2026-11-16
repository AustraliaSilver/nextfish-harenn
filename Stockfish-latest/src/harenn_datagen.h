#ifndef HARENN_DATAGEN_H_INCLUDED
#define HARENN_DATAGEN_H_INCLUDED

#include <filesystem>
#include <optional>
#include <sstream>

namespace Stockfish {

void run_harenn_datagen(std::istringstream& is, const std::optional<std::filesystem::path>& bin_path);

} // namespace Stockfish

#endif
