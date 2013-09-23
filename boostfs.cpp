#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <string>
#include <algorithm> // sort

static boost::filesystem::path outputDir;

std::vector<std::string> gatherMatchingFiles(std::string &target_path, std::string &pattern)
{
  pattern.insert(0, ".*");  // '.' is the wildcard in Perl regexp; '*' just means "repeat".
  pattern.append(".*\\.tif");

  const boost::regex my_filter(pattern);

  std::vector< std::string > all_matching_files;

  boost::filesystem::directory_iterator end_itr; // Constructs the end iterator.
  for( boost::filesystem::directory_iterator i( target_path ); i != end_itr; ++i ) {

    // Skip if not a file
    if( !boost::filesystem::is_regular_file( i->status() ) ) continue;

    boost::smatch what;

    // Skip if no match
    if( !boost::regex_match( i->path().string(), what, my_filter ) ) continue;

    // File matches, store it
    all_matching_files.push_back( i->path().string() );
  }

  // sort file names so that earlier time points will be processed first:
  sort(all_matching_files.begin(), all_matching_files.end());


  // Create output subfolder "decon/" just under the data folder:
  outputDir = target_path;
  outputDir /= "GPUdecon";

  if (! boost::filesystem::exists(outputDir) )
    boost::filesystem::create_directory(outputDir);

  return all_matching_files;
}


std::string makeOutputFilePath(std::string inputFileName, std::string insert)
{
  boost::filesystem::path inputpath(inputFileName);
  boost::filesystem::path outputpath(outputDir);

  std::string basename = inputpath.filename().string();
  int pos = basename.find_last_of(".tif");
  basename.insert(pos - 3, insert);

  outputpath /= basename;

  std::cout << "Output: " << outputpath.string() << '\n';
  return outputpath.string();
}
