/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>

#include "attacks.h"
#include "harenn_datagen.h"
#include "misc.h"
#include "tune.h"
#include "uci.h"

using namespace Stockfish;

#ifdef UNIVERSAL_BINARY
namespace Stockfish {

int main(int argc, char* argv[]);  // silence 'no previous declaration'

__attribute__((used)) // keep main alive
#endif

int main(int argc, char* argv[]) {
    std::cout << engine_info() << std::endl;

    Attacks::init();

    if (argc > 1 && std::string_view(argv[1]) == "harenn_datagen") {
        std::string cmd;
        for (int i = 2; i < argc; ++i) cmd += std::string(argv[i]) + " ";
        std::istringstream is(cmd);
        run_harenn_datagen(is, path_from_utf8(argv[0]));
        return 0;
    }

    auto cli = CommandLine(argc, argv);
    auto uci = std::make_unique<UCIEngine>(std::move(cli));

    Tune::init(uci->engine_options());

    uci->loop();

    return 0;
}

#ifdef UNIVERSAL_BINARY
}  // namespace Stockfish

    #ifdef UNIVERSAL_NEEDS_MAIN_SHIM
int main(int argc, char* argv[]) { return Stockfish::main(argc, argv); }
    #endif
#endif
