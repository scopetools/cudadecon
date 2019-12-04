#include "Buffer.h"

Buffer::~Buffer() noexcept(false) {
}

void Buffer::dump(std::ostream& s, int numCols)
{
  dump(s, numCols, 0, getSize());
}

