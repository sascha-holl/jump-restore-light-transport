#include "parsescene.h"
#include "pathtrace.h"
#include "mlt.h"
#include "image.h"
#include "camera.h"
#include "texturesystem.h"
#include "parallel.h"
#include "path.h"

#include <iostream>
#include <string>

// [sholl] added windows support
#ifdef _WIN32
#   include "unistd.h"
#else
#   include <unistd.h>
#   include <sys/resource.h>
#endif

// [sholl]
#include "sholl/integrator/metropolis_integrator.hpp"
#include "sholl/integrator/metropolis_restore_integrator.hpp"


void DptInit() {
    TextureSystem::Init();
    std::cout << "Langevin MCMC dpt version 1.0" << std::endl;
    std::cout << "Running with " << NumSystemCores() << " threads." << std::endl;
    if (getenv("DPT_LIBPATH") == nullptr) {
        std::cout
            << "Environment variable DPT_LIBPATH not set."
               "Please set it to a directory (e.g. export DPT_LIBPATH=/dir/to/dpt/src/bin) so "
               "that I can write/find the dynamic libraries." << std::endl;
        exit(0);
    }
}

void DptCleanup() {
    TextureSystem::Destroy();
    TerminateWorkerThreads();
}
 

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        return 0;
    }

    try {
        DptInit();

        bool compilePathLib = false;
        bool compileBidirPathLib = false;
        bool compileBidirPathLib2 = false;
        int maxDervDepth = 8;
        std::vector<std::string> filenames;
        int seedoffset = 0;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--compile-pathlib") {
                compilePathLib = true;
            } else if (std::string(argv[i]) == "--compile-bidirpathlib") {
                compileBidirPathLib = true;
            } else if (std::string(argv[i]) == "--compile-bidirpathlib2") {
                compileBidirPathLib2 = true; 
            } else if (std::string(argv[i]) == "--max-derivatives-depth") {
                maxDervDepth = std::stoi(std::string(argv[++i]));
            } else if (std::string(argv[i]) == "--seedoffset") {
                seedoffset = std::stoi(std::string(argv[++i]));
            }
            else {
                filenames.push_back(std::string(argv[i]));
            }
        }

        if (compilePathLib) {
            CompilePathFuncLibrary(false, maxDervDepth);
        }
        if (compileBidirPathLib) {
            CompilePathFuncLibrary(true, maxDervDepth);
        }
        if (compileBidirPathLib2) {
            CompilePathFuncLibrary2(maxDervDepth);
        }

        std::string cwd = getcwd(NULL, 0);
        for (std::string filename : filenames) {
            if (filename.rfind('/') != std::string::npos &&
                chdir(filename.substr(0, filename.rfind('/')).c_str()) != 0) {
                Error("chdir failed");
            }
            if (filename.rfind('/') != std::string::npos) {
                filename = filename.substr(filename.rfind('/') + 1);
            }

            std::unique_ptr<Scene> scene = ParseScene(filename);

            std::string integrator = scene->options->integrator;
                
            scene->options->seedOffset = seedoffset;

            // begin [sholl]
            if (integrator == "mc")
                PathTrace(scene.get(), BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth));
            else
            {
                std::shared_ptr<PathFuncLib const> const library = scene->options->mala ?
                    BuildPathFuncLibrary2(maxDervDepth) : // build first-order derivatives
                    BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth); // build first- and second-order derivatives

                if (integrator == "mcmc")
                    MLT(scene.get(), library);
                else if (integrator == "metropolis")
                {
                    if (scene->options->mala)
                        sholl::metropolis_integrator<LargeStep, MALASmallStep>{ scene.get(), library }.render();
                    else if (scene->options->h2mc)
                        sholl::metropolis_integrator<LargeStep, H2MCSmallStep>{ scene.get(), library }.render();
                    else
                        sholl::metropolis_integrator<LargeStep, SmallStep>{ scene.get(), library }.render();
                }
                else if (integrator == "metropolis_restore")
                {
                    if (scene->options->mala)
                        sholl::metropolis_restore_integrator<LargeStep, MALASmallStep>{ scene.get(), library }.render();
                    else if (scene->options->h2mc)
                        sholl::metropolis_restore_integrator<LargeStep, H2MCSmallStep>{ scene.get(), library }.render();
                    else
                        sholl::metropolis_restore_integrator<LargeStep, SmallStep>{ scene.get(), library }.render();
                }
                else
                    Error("Unknown integrator");
            }
            // end [sholl]
            
            if (chdir(cwd.c_str()) != 0) {
                Error("chdir failed");
            }
        }
        DptCleanup();
    } catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
