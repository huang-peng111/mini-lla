#ifndef MINI_LLA_TESTS_TEST_MAIN_H_
#define MINI_LLA_TESTS_TEST_MAIN_H_

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mini_lla {
namespace test {

// 返回 tiny 模型目录:优先用当前目录下的 models/tiny,其次退回上一级。
inline std::string TinyModelDir() {
  if (std::filesystem::exists("models/tiny/model.json")) return "models/tiny";
  if (std::filesystem::exists("../models/tiny/model.json")) {
    return "../models/tiny";
  }
  throw std::runtime_error("找不到 models/tiny,请在项目根目录运行测试");
}

// 返回临时目录下的一个测试文件路径。
inline std::string TempPath(const std::string& name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace test
}  // namespace mini_lla

// 断言条件成立,否则抛出异常并带上出错位置。
#define CHECK_TRUE(condition)                                              \
  do {                                                                     \
    if (!(condition)) {                                                    \
      throw std::runtime_error(std::string(__FILE__) + ":" +               \
                               std::to_string(__LINE__) +                  \
                               " CHECK_TRUE 失败: " #condition);           \
    }                                                                      \
  } while (false)

// 断言两个整数相等。
#define CHECK_EQUAL(actual, expected)                                      \
  do {                                                                     \
    const auto actual_value = static_cast<long long>(actual);               \
    const auto expected_value = static_cast<long long>(expected);           \
    if (actual_value != expected_value) {                                   \
      throw std::runtime_error(std::string(__FILE__) + ":" +                \
                               std::to_string(__LINE__) +                   \
                               " CHECK_EQUAL 失败: " #actual);              \
    }                                                                      \
  } while (false)

// 断言两个浮点数在容差内相等。
#define CHECK_NEAR(actual, expected, tolerance)                             \
  do {                                                                      \
    const double actual_value = (actual);                                   \
    const double expected_value = (expected);                               \
    if (std::fabs(actual_value - expected_value) > (tolerance)) {            \
      throw std::runtime_error(std::string(__FILE__) + ":" +                 \
                               std::to_string(__LINE__) +                    \
                               " CHECK_NEAR 失败: " #actual);                \
    }                                                                       \
  } while (false)

// 断言表达式会抛出 std::exception,用于错误路径。
#define CHECK_THROWS(expression)                                            \
  do {                                                                      \
    bool threw = false;                                                     \
    try {                                                                   \
      expression;                                                           \
    } catch (const std::exception&) {                                       \
      threw = true;                                                         \
    }                                                                       \
    if (!threw) {                                                           \
      throw std::runtime_error(std::string(__FILE__) + ":" +                \
                               std::to_string(__LINE__) +                   \
                               " CHECK_THROWS 失败: " #expression);         \
    }                                                                       \
  } while (false)

#endif  // MINI_LLA_TESTS_TEST_MAIN_H_
