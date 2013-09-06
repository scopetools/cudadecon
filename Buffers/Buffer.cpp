#include "Buffer.h"

Buffer::~Buffer() {
}

void Buffer::dump(std::ostream& s, int numCols)
{
  dump(s, numCols, 0, getSize());
}

