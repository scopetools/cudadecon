#include "Buffer.h"
#include "CPUBuffer.h"
#include "GPUBuffer.h"
#include "gtest/gtest.h"
#include <cstdlib>


int compareArrays(char* arr1, char* arr2, int size);
TEST(CPUBuffer, IncludeTest) {
  ASSERT_EQ(0, 0);
}
TEST(CPUBuffer, ConstructorTest) {
  CPUBuffer a;
  ASSERT_EQ(0, a.getSize());
  ASSERT_EQ(0, a.getPtr());
  CPUBuffer b(4 * sizeof(float));
  ASSERT_EQ(4 * sizeof(float), b.getSize());
  EXPECT_TRUE(0 != b.getPtr());
}
TEST(CPUBuffer, ResizeTest) {
  CPUBuffer a;
  a.resize(10);
  ASSERT_EQ(10, a.getSize());
  EXPECT_TRUE(0 != a.getPtr());
}
TEST(CPUBuffer, SetFromPlainArrayTest) {
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
}
TEST(CPUBuffer, SetTest) {
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

  CPUBuffer b;
  b.resize(a.getSize());
  a.set(&b, 0, 4 * sizeof(float), 0);
  b.setPlainArray(out, 0, b.getSize(), 0);
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result, sizeof(result)));
}
TEST(CPUBuffer, TakeOwnershipTest) {
  CPUBuffer a;
  float* src = new float[4];
  src[0] = 11.0;
  src[1] = 22.0;
  src[2] = 33.0;
  src[3] = 44.0;
  float result[4] = {11.0, 22.0, 33.0, 44.0};
  a.takeOwnership(src,  4 * sizeof(float));
  ASSERT_EQ(0,
      compareArrays((char*)a.getPtr(), (char*)result, sizeof(result)));
}
TEST(CPUBuffer, Dump) {
  CPUBuffer a;
  float* src = new float[4];
  src[0] = 11.0;
  src[1] = 22.0;
  src[2] = 33.0;
  src[3] = 44.0;
  a.takeOwnership(src,  4 * sizeof(float));
  a.dump(std::cout, 2);
  ASSERT_EQ(0, 0);
}
TEST(CPUBuffer, GPUSetTest) {
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

  CPUBuffer c;
  c.resize(b.getSize());
  ASSERT_EQ(4 * sizeof(float), c.getSize());
  b.set(&c, 0, 4 * sizeof(float), 0);
  c.setPlainArray(out, 0, c.getSize(), 0);
  ASSERT_EQ(0,
      compareArrays((char*)out, (char*)result, sizeof(result)));
}

int compareArrays(char* arr1, char* arr2, int size) {
  int difference = 0;
  for (int i = 0; i < size; ++i) {
    difference += abs(arr1[i] - arr2[i]);
  }
  return difference;
}

