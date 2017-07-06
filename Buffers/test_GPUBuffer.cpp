#include "gtest/gtest.h"
#include "Buffer.h"
#include "GPUBuffer.h"
#include "CPUBuffer.h"
#include <cstdlib>


int compareArrays(char* arr1, char* arr2, int size);

TEST(GPUBuffer, IncludeTest) {
  ASSERT_EQ(0, 0);
}
TEST(GPUBuffer, ConstructorTest) {
  GPUBuffer a;
  ASSERT_EQ(0, a.getSize());
  ASSERT_EQ(0, a.getPtr());
  GPUBuffer b(4 * sizeof(float), 0);
  ASSERT_EQ(4 * sizeof(float), b.getSize());
  EXPECT_TRUE(0 != b.getPtr());

  GPUBuffer c;
  GPUBuffer d(10, 0);
  c = d;
}
TEST(GPUBuffer, ResizeTest) {
  GPUBuffer a(0);
  a.resize(10);
  ASSERT_EQ(10, a.getSize());
  EXPECT_TRUE(0 != a.getPtr());
}
TEST(GPUBuffer, GPUSetTest) {
  CPUBuffer a;
  ASSERT_EQ(0, a.getSize());
  ASSERT_EQ(0, a.getPtr());
  a.resize(4 * sizeof(float));
  float src[4] = {11.0, 22.0, 33.0, 44.0};
  float result[4] = {11.0, 22.0, 33.0, 44.0};
  float out[4];
  a.setFrom(src, 0, sizeof(src), 0);
  a.setPlainArray(out, 0, a.getSize(), 0);
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result, sizeof(result)));

  GPUBuffer b;
  b.resize(a.getSize());
  a.set(&b, 0, 4 * sizeof(float), 0);

  GPUBuffer c;
  c.resize(b.getSize());
  ASSERT_EQ(4 * sizeof(float), c.getSize());
  b.set(&c, 0, 4 * sizeof(float), 0);

  CPUBuffer d;
  d.resize(c.getSize());
  ASSERT_EQ(4 * sizeof(float), d.getSize());
  c.set(&d, 0, c.getSize(), 0);
  d.setPlainArray(out, 0, d.getSize(), 0);
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result, sizeof(result)));

  a.set(&c, 0, 2 * sizeof(float), 2 * sizeof(float));
  c.dump(std::cout, 2);
  c.set(&d, 0, c.getSize(), 0);
  d.setPlainArray(out, 0, d.getSize(), 0);
  float result2[] = {11.0, 22.0, 11.0, 22.0};
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result2, sizeof(result2)));

}
TEST(GPUBuffer, DumpTest) {
  CPUBuffer a;
  ASSERT_EQ(0, a.getSize());
  ASSERT_EQ(0, a.getPtr());
  a.resize(4 * sizeof(float));
  float src[4] = {11.0, 22.0, 33.0, 44.0};
  float result[4] = {11.0, 22.0, 33.0, 44.0};
  float out[4];
  a.setFrom(src, 0, sizeof(src), 0);
  a.setPlainArray(out, 0, a.getSize(), 0);
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result, sizeof(result)));

  GPUBuffer b;
  b.resize(a.getSize());
  a.set(&b, 0, 4 * sizeof(float), 0);

  GPUBuffer c;
  c.resize(b.getSize());
  ASSERT_EQ(4 * sizeof(float), c.getSize());
  b.set(&c, 0, 4 * sizeof(float), 0);
  c.dump(std::cout, 2);
}

int compareArrays(char* arr1, char* arr2, int size) {
  int difference = 0;
  for (int i = 0; i < size; ++i) {
    difference += abs(arr1[i] - arr2[i]);
  }
  return difference;
}

