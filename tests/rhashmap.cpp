#include <crhashtable.hpp>
#include <format>
#include <cstdio>

using namespace jh;

/**
 * Compile test of RHashMap.
 * Some runtime checks are also done to satisfy the nodiscard attribute.
 */
constexpr int CHECK_V = 999;
struct check
{
    int v = CHECK_V;
};

using MapType = RHashMap<int, check>;

int compare_nro = 1;

template<typename Result, typename Expected>
void test_eq(const Result & a, const Expected b, const char* func, int line)
{
    if (a != b) {
        std::printf("FAIL: %s at line: %d", func, line);
        std::exit(compare_nro);
    }
    ++compare_nro;
}

template<typename Result, typename Expected>
void test_neq(const Result & a, const Expected b, const char* func, int line)
{
    if (a == b) {
        std::printf("FAIL: %s at line: %d", func, line);
        std::exit(compare_nro);
    }
    ++compare_nro;
}

#define compare_eq(A, B) test_eq((A), (B), __func__, __LINE__)
#define compare_neq(A, B) test_neq((A), (B), __func__, __LINE__)

void test_contruct()
{
    MapType map;
    auto copy_empty = map;
    auto move_empty = std::move(map);
    map = MapType(1024);
}

void test_emplace()
{
    MapType map;
    auto v = map.emplace(0, check{});
    static_assert(!std::is_const_v<decltype(v.first)>);
    static_assert(std::is_const_v<decltype(v.first->first)>);
    static_assert(!std::is_const_v<decltype(v.first->second)>);
    compare_eq(v.second, true);
    int local_x = 1;
    check local_c;
    const int local_cx = 2;
    const check local_cc{};
    v = map.emplace(local_x, check{});
    v = map.emplace(local_cx, local_c);
    v = map.emplace(local_cx, local_cc);
    compare_eq(v.second, false);
    v = map.emplace(local_x, std::move(local_c));
}

void test_find_erase()
{
    MapType map;
    for (int i = 0; i < 10; ++i)
        map.emplace(i, check{});
    // auto find(F&& kval) const -> const_local_iterator
    auto ci = std::as_const(map).find(0);
    static_assert(std::is_same_v<decltype(ci), MapType::const_local_iterator>);
    // auto find(F&& kval) -> local_iterator
    auto i = map.find(0);
    static_assert(std::is_same_v<decltype(i), MapType::local_iterator>);
    // auto erase(local_iterator rm) -> local_iterator
    map.erase(i);
    // auto erase(value_iterator rm) -> value_iterator
    map.erase(map.begin());
    // auto erase(const Key& kval) -> size_type
    compare_eq(map.erase(0), 0);
    compare_eq(map.erase(2), 1);
    map = MapType();
    compare_eq(map.erase(7), 0);
}

void test_contains_count_empty()
{
    MapType map;
    compare_eq(map.count(0), 0);
    compare_eq(map.empty(), true);
    map.emplace(0, check{});
    compare_eq(map.contains(0), true);
    int key = 1;
    compare_eq(map.contains(key), false);
    compare_eq(map.count(0), 1);
    compare_eq(map.empty(), false);
}

void test_iterators()
{
    MapType map;
    compare_eq(map.begin(), map.end());
    compare_eq(map.begin(), map.vend());
    compare_eq(map.cbegin(), map.cend());
    compare_eq(map.cbegin(), map.vcend());
    compare_eq(std::as_const(map).begin(), std::as_const(map).end());
    compare_eq(std::as_const(map).begin(), std::as_const(map).vend());
    compare_eq(std::as_const(map).cbegin(), std::as_const(map).cend());
    compare_eq(std::as_const(map).cbegin(), std::as_const(map).vcend());

    for (int i = 0; i < 10; ++i) {
        auto v = map.emplace(i, check{i});
        compare_neq(v.first, map.end());
        compare_neq(v.first, map.vend());
    }

    for (auto& k : map) { (volatile void)k; }
    for (auto& k : std::as_const(map)) { (volatile void)k; }
    for (auto& k : vrange(map)) { (volatile void)k; }
    for (auto& k : vrange(std::as_const(map))) { (volatile void)k; }
    for (auto slot = map.local_begin(); slot != map.end(); ++slot) {
        (volatile void)slot;
    }

    MapType::value_iterator def;
    auto pm = map.begin();
    // friend bool operator!=(const ValueIterator& a, const ValueIterator& b)
    // reference operator*()
    auto v = *pm;
    static_assert(std::is_const_v<decltype(v.first)>);
    compare_eq((*pm).first, v.first);
    // pointer operator->()
    compare_eq(pm->second.v, 0);
    // const pointer operator->() const
    compare_eq(std::as_const(pm)->second.v, 0);
    // auto operator++(int) -> auto
    compare_eq(pm++, map.begin());
    // auto operator++() -> auto&
    ++pm;
    // friend bool operator==(const ValueIterator& a, const ValueIterator& b)
    compare_neq(pm, map.begin());
    // auto operator--(int) -> auto
    compare_neq(pm--, map.begin());
    // auto operator--() -> auto&
    --pm;
    compare_eq(pm, map.begin());
    // reference operator[] (difference_type n) const
    using DiffType = MapType::value_iterator::difference_type;
    compare_eq(pm[DiffType(2)].first, 2);
    // auto operator+= (difference_type n) -> auto&
    pm += DiffType(2);
    // auto operator+ (difference_type n) const -> auto
    compare_neq(pm + DiffType(2), pm);
    // auto operator-= (difference_type n) -> auto&
    pm -= DiffType(2);
    // auto operator- (difference_type n) const -> auto
    compare_neq(pm - DiffType(2), pm);
    // difference_type operator- (const ValueIterator& lhs, const ValueIterator& rhs)
    compare_eq(std::distance(map.begin(), map.vend()), map.size());
    compare_eq(std::distance(map.vend(), map.begin()), -map.size());

    // operator local_iterator() const
    auto l = static_cast<MapType::local_iterator>(map.begin());
    // note: no erasures.
    compare_eq(l, pm);
}

void test_misc()
{
    MapType map;
    for (int i = 0; i < 10; ++i) {
        map.emplace(i, check{i});
    }

    const auto num_buckets = map.bucket_count();
    auto lf = map.load_factor();
    (volatile void)lf;
    auto eq = map.key_eq();
    (volatile void)eq;
    auto hash = map.hash_function();
    (volatile void)hash;

    map.clear();
    compare_eq(map.empty(), true);
    compare_eq(map.size(), 0);
    compare_eq(map.bucket_count(), num_buckets);

    map.shrink_to_fit();
    compare_neq(map.bucket_count(), num_buckets);

    auto inv = map.end();
    map.rehash(1024);
    compare_neq(inv, map.end());
}

int main()
{
    // HashtableAlgorithms() =default;
    // HashtableAlgorithms(HashtableAlgorithms && cont) noexcept
    // HashtableAlgorithms(const HashtableAlgorithms & cont)
    // HashtableAlgorithms(size_t max_elements, const Hasher& hash = Hasher(), const Equal& equal = Equal())
    // ~HashtableAlgorithms()
    // auto operator= (const HashtableAlgorithms& cont) -> auto&
    // auto operator= (HashtableAlgorithms&& cont) noexcept -> auto&
    test_contruct();
    //
    // template <typename F, typename... Args>
    // auto emplace(F&& key, Args&&... args) -> std::pair<local_iterator, bool>
    test_emplace();
    // auto erase(local_iterator rm) -> local_iterator
    // auto erase(const Key& kval) -> size_type
    // auto erase(value_iterator rm) -> value_iterator
    // auto find(F&& kval) -> local_iterator
    // auto find(F&& kval) const -> const_local_iterator
    test_find_erase();
    // auto contains(const Key& kval) const
    // size_type count(const Key& kval) const
    // auto empty() const
    test_contains_count_empty();
    //
    // auto begin()
    // auto begin() const
    // auto cbegin() const
    // auto end()
    // auto end() const
    // auto cend() const
    // auto vend()
    // auto vend() const
    // auto vcend() const
    // vrange()
    test_iterators();

    // void clear()
    // void shrink_to_fit()
    // void max_load_factor(float lf)
    // float max_load_factor() const
    // float load_factor() const
    // void rehash(size_t new_size)
    // size_type bucket_count() const noexcept
    // size_type size() const noexcept
    // auto hash_function() const
    // auto key_eq() const
    test_misc();
    return 0;
}
