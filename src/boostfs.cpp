#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <regex>
#include <iostream>
#include <string>
#include <algorithm> // sort

#ifdef _WIN32
#include <windows.h>
#endif

static boost::filesystem::path dataDir;


std::vector<std::string> gatherMatchingFiles(std::string &target_path, std::string &pattern, bool no_overwrite)
{
#ifdef _WIN32
  HANDLE  hConsole;
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif
	//std::cout << std::endl;
	
	//***************** make GPUdecon output folder ***************

  // Create output subfolder "GPUdecon/" just under the data folder:
  dataDir = target_path;
  boost::filesystem::path outputDir;
  outputDir = dataDir / "GPUdecon";

  //***************** make regex filter ***************
  // Check if the pattern is specified as a full file name; if so, insert
  // an escape char '\' before '.'. Necessary?
  size_t p1 = pattern.rfind(".tif");
  size_t p2 = pattern.rfind(".TIF");
  if (p1 != std::string::npos || p2 != std::string::npos) {
    // do not append ".*tif" at the end of pattern
  }
  else 
    pattern.append(".*\\.[tT][iI][fF]");
  pattern.insert(0, ".*");  // '.' is the wildcard in Perl regexp; '*' just means "repeat".
  std::cout << "file regex pattern: " << pattern << std::endl;

  const std::regex my_filter(pattern);


  //***************** make vector of files we have already created ***************
  std::vector< std::string > all_decon_files;

  if ( boost::filesystem::exists(outputDir) ){

    boost::filesystem::directory_iterator end_decon_itr; // Constructs the end iterator.
    for (boost::filesystem::directory_iterator i(outputDir); i != end_decon_itr; ++i){

  	  // Skip if not a file. This won't be a match.
  	  if (!boost::filesystem::is_regular_file(i->status())) continue;

      // Skip if no match.  This won't be a match.
      std::smatch what;
      std::string tmps=i->path().string(); // regex_match no longer permits temporary strings 
      if (!std::regex_match(tmps, what, my_filter)) continue;

  	  all_decon_files.push_back(i->path().stem().string());
    }

  }

  //*********************************************

  std::vector< std::string > all_matching_files;

  int outer_loop = 0; // outer loop counter
  
  boost::filesystem::directory_iterator end_itr; // Constructs the end iterator.
  for( boost::filesystem::directory_iterator i( target_path ); i != end_itr; ++i ) { // loop on all files in the input folder

	  outer_loop++;
    // Skip if not a file
    if( !boost::filesystem::is_regular_file( i->status() ) ) continue;

    std::smatch what;

    // Skip if no match
    std::string tmps=i->path().string(); // regex_match no longer permits temporary strings 
    if( !std::regex_match( tmps, what, my_filter ) ) continue;

	// Skip if we are not "overwriting" and if the file exists in the output folder already
	if (no_overwrite)
	{
		//SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
		
		bool my_match = false; // did we already deconvolve it?
		int loop = 0;	// inner loop counter

		std::string my_string; // the file stem from the decon folder
		std::string my_prefix; // the file stem from the raw data folder

		for (auto & element : all_decon_files) // loop through all files in the all_decon_files vector
		{
			if (my_match == false){

				loop++;

				my_string = element;
				my_prefix = i->path().stem().string();

				my_match = boost::algorithm::starts_with(my_string, my_prefix); //check if the original filename  (excluding extension) matches the beginning of the decon name

				if (my_string.find(my_prefix) == 0) // Check a second way.
					my_match = true;

				}
		}
		//SetConsoleTextAttribute(hConsole, my_match ? 15 : 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
		//std::cout << my_prefix << (my_match ? "   Yes " : "No     ") << loop << "  " << outer_loop << "  " << std::endl << my_string << std::endl;
		if (my_match) continue;

	}


    // File matches, store it
    all_matching_files.push_back( i->path().string() ); // this statement is skipped if "continue" is executed
  }

  // sort file names so that earlier time points will be processed first:
  sort(all_matching_files.begin(), all_matching_files.end());




  return all_matching_files;
}

void makeNewDir(std::string subdirname)
{
  boost::filesystem::path outputDir = dataDir/subdirname;
  if (! boost::filesystem::exists(outputDir) )
    boost::filesystem::create_directory(outputDir);
}

std::string makeOutputFilePath(std::string inputFileName, std::string subdir, std::string insert)
{
  boost::filesystem::path inputpath(inputFileName);
  boost::filesystem::path outputpath(dataDir/subdir);

  std::string basename = inputpath.filename().string();
  int pos = basename.find_last_of(".tif");
  basename.insert(pos - 3, insert);

  outputpath /= basename;

  std::cout << "Output: " << outputpath.string() << '\n';
  return outputpath.string();
}
