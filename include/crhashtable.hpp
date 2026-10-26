// SPDX-License-Identifier: BSL-1.0
// Copyright (C) 2026 Jarmo A Tiitto <jarmo.tiitto@gmail.com>
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef CRHASHTABLE_43FF74E8499_H_INCLUDED
#define CRHASHTABLE_43FF74E8499_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

/**
 * Hashtable by JATothrim.
 *
 * - Uses robin-hood hashing.
 * - Contiguous element data. (see caveat)
 * - 64-bit metadata per element: 16-bit distance-from-home, 16-bit hash fingerprint and 32-bit index.
 * - Elements can be iterated in insertion order until ERASE().
 *
 * 64-bit hash function is required. Non-avalanching or lower than 64 bits can severely disable performance.
 *
 * The table uses element indexes into the actual element container, such that robin-hood induced movement
 * of the elements themselves is nearly eliminated. When erasing element, the hash-value is computed for
 * the last element in the array and the slot of this at-end element is searched to do an ERASE(). This
 * element is moved into the erased element, and the metadata is updated. ERASE() operation therefore
 * invalidates all references and iterators to elements.
 *
 * The container uses an local_iterator type for emplace(), find(), erase() and end(). Range-for loops
 * should always use the jh::vrange() helper to use the vend() instead of end() because of this. The
 * local_iterator is not optimal for iterating through the container, or repeated access, since it iterates
 * over metadata slots. The local_iterator is however efficiently convertible to a value_iterator that
 * provides iteration over the container's element array. The past-end local_iterator and value_iterator
 * compare equal.
 *
 * The begin() returns a value_iterator that provides fast iteration over all elements. A value_iterator
 * can be converted back into a local_iterator. This however requires finding the metadata slot of the
 * pointed-to element, but such a conversion is still more efficient than defererencing the value_iterator
 * and using find().
 *
 * RHashMap< Key, Mapped> and RHashSet< Key> variants of the container are provided. The key_type cannot be
 * mutated once inserted: The map variant value_type is std::pair< const Key, Mapped>. The set variant
 * key_type and value_type are const Key. The value_type must be moveable.
 *
 */

namespace jh {

/**
 * Declare primary container template.
 */
template <typename, typename, typename, typename>
class HashTableAlgorithms;

/**
 * Bucket iterator template.
 */
template<typename, bool = false>
class BucketIterator;

/**
 * Value iterator template.
 */
template<typename, bool = false>
class ValueIterator;

/**
 * Map vs. set container API
 */

template<typename Key, typename Mapped>
class Config {
protected:
	using element_type = std::pair<const Key, Mapped>;
	using store_type = element_type;

	static constexpr auto to_key(const element_type& value) -> auto
	{
		return const_cast<const Key&>(value.first);
	}

	static constexpr auto to_key(const Key& value) -> auto
	{
		return value;
	}

	static constexpr void move_assign(element_type& dst, element_type& src)
	{
		const_cast<Key&>(dst.first) = std::move(const_cast<Key&>(src.first));
		dst.second = std::move(src.second);
	}

public:
	// Public API typedefs.
	using key_type = const Key;
	using mapped_type = Mapped;
	using value_type = element_type;
};

template<typename Key>
class Config<Key, void> {
protected:
	using element_type = const Key;
	using store_type = Key;

	static constexpr auto to_key(const Key& value) -> auto
	{
		return value;
	}

	static constexpr void move_assign(store_type& dst, store_type& src)
	{
		dst = std::move(src);
	}

public:
	// Public API typedefs.
	using key_type = element_type;
	using value_type = element_type;
};

/**
 * Hashtable algorithms.
 * @brief the implementation of RHash[Set,Map]
 */
template <typename Key, typename Value, typename Hasher, typename Equal>
class HashtableAlgorithms : public Config<Key, Value> {
protected:

	using hasher_t = Hasher;
	using equal_t = Equal;

	friend class BucketIterator<HashtableAlgorithms>;
	friend class BucketIterator<HashtableAlgorithms, true>;
	friend class ValueIterator<HashtableAlgorithms>;
	friend class ValueIterator<HashtableAlgorithms, true>;

	using slot_iterator = size_t;

	struct meta_t {
		uint32_t dfh_hash;
		uint32_t nth;
	};
	using slot_type = decltype(std::declval<meta_t>().dfh_hash);
	using index_type = decltype(std::declval<meta_t>().nth);

public:
	using local_iterator = BucketIterator<HashtableAlgorithms>;
	using value_iterator = ValueIterator<HashtableAlgorithms>;
	using const_local_iterator = BucketIterator<HashtableAlgorithms, true>;
	using const_value_iterator = ValueIterator<HashtableAlgorithms, true>;

private:
	static constexpr slot_type SLOT_BITS = 8 * sizeof(slot_type);
	// Count of DFH bits.
	static constexpr uint32_t DFH_SHIFT = SLOT_BITS / 2;

	// Note: the members are ordered to minimize table size.

	// Actual element data.
	using value_t = typename Config<Key, Value>::element_type;
	using elements_t = std::vector<typename Config<Key, Value>::store_type>;
	elements_t m_elements;
	// Metadata table.
	meta_t* m_metatable = nullptr;
	const meta_t* m_metatable_end = nullptr;
	// </cacheline>
	// 2^shift capacity.
	uint32_t m_shift = 0;
	// Max count of elements before an rehash takes place.
	size_t m_max_fill = 0;
	size_t m_max_load_factor = 800; // 80.0%, 800/1000
	hasher_t m_hasher;
	equal_t m_equal;

protected:

	/**
	 * Metadata access.
	 */

	[[nodiscard]] inline auto mcapacity() const noexcept -> auto
	{
		return size_t(1) << m_shift;
	}

	[[nodiscard]] inline auto modulus(size_t hash_value) const noexcept -> auto
	{
		// The low hash bits become part of the slot's probe hash.
		// These bits shouldn't be part of the modulus, thus the modulus
		// is computed via right-shifting the hash bits into the correct range.
		return hash_value >> (sizeof(hash_value) * 8 - m_shift);
	}

	[[nodiscard]] inline auto at_modulus(size_t hash_value) noexcept -> auto
	{
		return m_metatable + modulus(hash_value);
	}

	[[nodiscard]] inline auto at_modulus(size_t hash_value) const noexcept -> const auto*
	{
		return m_metatable + modulus(hash_value);
	}

	[[nodiscard]] inline auto const_at_modulus(size_t hash_value) const noexcept -> const auto*
	{
		return m_metatable + modulus(hash_value);
	}

	// Distance-from-home
	// Zero is reserved for empty slot so 1 is at home slot.
	static constexpr auto dfh(slot_type x) noexcept
	{
		return x >> (SLOT_BITS - DFH_SHIFT);
	}

	// The DFH bits are arranged such that less-than comparison of the whole slot_type
	// value works regardless of what the probe hash bits are.
	static constexpr auto dfh_increment() noexcept
	{
		return slot_type(1) << (SLOT_BITS - DFH_SHIFT);
	}

	// Can dfh_hash be incremented by dfh_increment()?
	static constexpr auto dfh_overflow(slot_type dfh_hash) noexcept
	{
		return dfh_hash >= ((~slot_type(0)) << (SLOT_BITS - DFH_SHIFT));
	}

	static constexpr auto probe_hash(size_t hash_value) noexcept
	{
		// Grab the low DFH_SHIFT bits.
		return slot_type(hash_value & ((slot_type(1) << DFH_SHIFT) - 1));
	}

	// Advance to next bucket or wraparound.
	template<typename Meta>
	inline void next_bucket(Meta*& ptr) const noexcept
	{
		++ptr;
		if (ptr == m_metatable_end) {
			ptr = m_metatable;
		}
	}

	[[nodiscard]] inline auto size_limit() const noexcept
	{
		return m_max_fill;
	}

	[[nodiscard]] inline auto size_limit_low() const noexcept
	{
		return m_max_fill / 2;
	}

	[[nodiscard]] inline auto max_fill_from_pow2(size_t pow2) const noexcept
	{
		// Floating point would be easier, but this would sprinkle the already bulky emplace()
		// with FPU instructions. Thus, do fixed-point computation here.
		return ((pow2 * m_max_load_factor) % (pow2 * 1000)) / 1000;
	}

	[[nodiscard]] auto calculate_shift_for(size_t max_elements) const noexcept
	{
		int shift = 1;
		size_t pow2 = 2;
		while (max_fill_from_pow2(pow2) < max_elements) {
			pow2 *= 2;
			++shift;
		}
		return shift;
	}


	void grow_and_clear()
	{
		// Grow capacity and throw away the metadata.
		if (!m_shift) {
			m_shift = calculate_shift_for(8);
		} else {
			++m_shift;
		}
		m_max_fill = max_fill_from_pow2(mcapacity());

		// Delete first and then allocate to minimize peak memory usage.
		delete[] m_metatable;
		m_metatable = new meta_t[mcapacity()];
		std::memset(m_metatable, 0, sizeof(meta_t) * mcapacity());
		m_metatable_end = m_metatable + mcapacity();
	}

	void resize_and_clear(size_t max_elements)
	{
		// Set capacity and throw away the metadata.
		m_shift = calculate_shift_for(max_elements);
		m_max_fill = max_fill_from_pow2(mcapacity());

		delete[] m_metatable;
		m_metatable = new meta_t[mcapacity()];
		std::memset(m_metatable, 0, sizeof(meta_t) * mcapacity());
		m_metatable_end = m_metatable + mcapacity();
	}

	inline auto insert_index(size_t hash, index_type elementIndex) noexcept -> const auto*
	{
		// Metadata entry to be inserted:
		// DFH == 1 => element is at it's home index
		meta_t curr = {dfh_increment() + probe_hash(hash), elementIndex};

		// Find insertion point, i.e. skip over too poor slots.
		auto* bucket = at_modulus(hash);
		while (curr.dfh_hash <= bucket->dfh_hash) {
			assert(!dfh_overflow(curr.dfh_hash));
			curr.dfh_hash += dfh_increment();
			this->next_bucket(bucket);
		}

		auto* const element_bucket = bucket;

		// Rich slots are shifted forwards to make space for the poor.
		for(;;) {
			std::swap(*bucket, curr);
			if (!curr.dfh_hash) {
				return element_bucket;
			} else if (dfh_overflow(curr.dfh_hash)) [[unlikely]] {
				// ran out of DFH bits, the metadata now trash.
				return m_metatable_end;
			}
			curr.dfh_hash += dfh_increment();
			this->next_bucket(bucket);
		}
	}

	[[nodiscard]] inline auto find_index(size_t hash, uint32_t elementIndex) const noexcept -> auto*
	{
		const auto* bucket = const_at_modulus(hash);
		// The element must be at bucket or some slot after this.
		// A empty-slot cannot exist before the element's slot.
		while (bucket->nth != elementIndex) {
			assert(bucket->dfh_hash);
			this->next_bucket(bucket);
		}
		assert(bucket->dfh_hash);
		return const_cast<meta_t*>(bucket);
	}

	inline void erase_bucket(meta_t* bucket) noexcept
	{
		// Maintain the robin-hood hashing invariant:
		// Shift backwards from bucket+1 until empty slot or home slot is found.
		// The DFH of each moved slot consequently decremented by one.
		auto* prev = bucket;
		this->next_bucket(bucket);

		while (bucket->dfh_hash >= dfh_increment() * 2) {
			prev->dfh_hash = bucket->dfh_hash - dfh_increment();
			prev->nth = bucket->nth;
			prev = bucket;
			this->next_bucket(bucket);
		}
		*prev = {0, 0};
	}

	/**
	 * Element access.
	 */

	auto element(const meta_t* pos) -> auto*
	{
		if (pos == m_metatable_end) {
			return m_elements.data() + m_elements.size();
		}
		return &m_elements[pos->nth];
	}

	auto element(const meta_t* pos) const -> const auto*
	{
		if (pos == m_metatable_end) {
			return m_elements.data() + m_elements.size();
		}
		return &m_elements[pos->nth];
	}

	auto convert_iterator(value_t* itr) -> const auto*
	{
		if (itr == m_elements.data() + m_elements.size()) {
			return m_metatable_end;
		}
		auto hash = m_hasher(this->to_key(*itr));
		index_type pos = std::distance(m_elements.data(), itr);
		return find_index(hash, pos);
	}

	auto convert_iterator(const value_t* itr) const -> const auto*
	{
		if (itr == m_elements.data() + m_elements.size()) {
			return m_metatable_end;
		}
		auto hash = m_hasher(this->to_key(*itr));
		index_type pos = std::distance(m_elements.data(), itr);
		return find_index(hash, pos);
	}

	auto advance(const meta_t* itr, ptrdiff_t n) const -> const auto*
	{
		// Advance up to end(), skipping empty buckets.
		while (n && itr != m_metatable_end) {
			n -= itr->dfh_hash != slot_type(0);
			++itr;
		}
		return itr;
	}

	/**
	 * Rehash and re-insert all elements into the Metatable.
	 * @pre metadata table must have been cleared.
	 */
	auto do_rehash() -> const auto*
	{
		auto last = m_metatable_end;

		// Re-build metadata.
		for (size_t i = 0; i < m_elements.size(); ++i) {
			last = insert_index(m_hasher(this->to_key(m_elements[i])), i);
			if (last == m_metatable_end) {
				// Oops, too many collisions.
				grow_and_clear();
				i = 0;
			}
		}
		return last;
	}

	/**
	 * INSERT() a value, rehash if necessary.
	 */
	template <typename Item, typename... Args>
	auto do_emplace(const size_t hash, Item&& item, Args&&... args) -> const auto*
	{
		m_elements.emplace_back(std::forward<Item>(item), std::forward<Args>(args)...);

		if (m_elements.size() >= size_limit()) [[unlikely]] {
			// load-factor is too high => rehash or no metatable is allocated yet.
			grow_and_clear();
			return do_rehash();
		}

		auto itr = insert_index(hash, m_elements.size() - 1);
		if (itr == m_metatable_end) {
			grow_and_clear();
			return do_rehash();
		}
		return itr;
	}

	/**
	 * FIND()
	 */
	[[nodiscard]] auto do_find(const size_t hash, const Key& kval) const -> const auto*
	{
		slot_type dist = dfh_increment() + probe_hash(hash);
		const auto* bucket = const_at_modulus(hash);

		for (;;) {
			if (bucket->dfh_hash < dist) {
				// DFH is greater than inserted key can have => key not found.
				// This branch is before the below branch, to early return on empty home slots.
				return m_metatable_end;
			} else if (bucket->dfh_hash == dist) {
				if (m_equal(kval, this->to_key(m_elements[bucket->nth]))) {
					// Found!
					return bucket;
				}
			}
			this->next_bucket(bucket);
			dist += dfh_increment();
		}
	}

	/**
	 * ERASE()
	 */
	void do_erase(meta_t* itr)
	{
		const auto index = itr->nth;
		const slot_iterator at_end = m_elements.size() - 1;

		if (index != at_end) [[likely]] {
			// Erase the element: find where the m_elements.back() is in the metadata table,
			// and redirect it to itr's slot.
			const auto end_hash = m_hasher(this->to_key(m_elements.back()));
			find_index(end_hash, at_end)->nth = index;
			this->move_assign(m_elements[index], m_elements.back());
		}

		// Pop-back.
		m_elements.pop_back();
		erase_bucket(itr);
	}

	void swap(HashtableAlgorithms & cont) noexcept
	{
		using std::swap;
		std::swap(m_elements, cont.m_elements);
		std::swap(m_metatable, cont.m_metatable);
		std::swap(m_metatable_end, cont.m_metatable_end);
		std::swap(m_shift, cont.m_shift);
		std::swap(m_max_fill, cont.m_max_fill);
		std::swap(m_max_load_factor, cont.m_max_load_factor);
		std::swap(m_hasher, cont.m_hasher);
		std::swap(m_equal, cont.m_equal);
	}

public:

    using hasher = hasher_t;
    using key_equal = equal_t;
    // todo: using allocator_type = ;
	using size_type = std::size_t;

	HashtableAlgorithms() =default;

	HashtableAlgorithms(HashtableAlgorithms && cont) noexcept
		: m_elements(std::move(cont.m_elements))
		, m_metatable(std::exchange(cont.m_metatable, nullptr))
		, m_metatable_end(std::exchange(cont.m_metatable_end, nullptr))
		, m_shift(std::exchange(cont.m_shift, 0))
		, m_max_fill(std::exchange(cont.m_max_fill, 0))
		, m_max_load_factor(std::exchange(cont.m_max_load_factor, 0))
		, m_hasher(std::move(cont.m_hasher))
		, m_equal(std::move(cont.m_equal))
	{}

	HashtableAlgorithms(const HashtableAlgorithms & cont)
		: m_elements(cont.m_elements)
		, m_shift(cont.m_shift)
		, m_max_fill(cont.m_max_fill)
		, m_max_load_factor(cont.m_max_load_factor)
		, m_hasher(cont.m_hasher)
		, m_equal(cont.m_equal)
	{
		if (cont.m_metatable) {
			m_metatable = new meta_t[cont.mcapacity()];
			m_metatable_end = m_metatable + cont.mcapacity();
			std::copy_n(cont.m_metatable, cont.mcapacity(), m_metatable);
		}
	}

	auto operator= (const HashtableAlgorithms& cont) -> auto&
	{
		HashtableAlgorithms(cont).swap(*this);
		return *this;
	}

	auto operator= (HashtableAlgorithms&& cont) noexcept -> auto&
	{
		HashtableAlgorithms(std::move(cont)).swap(*this);
		return *this;
	}

	HashtableAlgorithms(size_t max_elements, const Hasher& hash = Hasher(), const Equal& equal = Equal())
		: m_hasher(hash)
		, m_equal(equal)
	{
		rehash(max_elements);
	}

	~HashtableAlgorithms()
	{
		delete[] m_metatable;
	}

	/**
	 * Construct value_type from @p key and pass additional arguments to the value_type constructor.
	 * @return local_iterator to the inserted element or to a existing element if the key already exists.
	 */
	template <typename F, typename... Args>
	auto emplace(F&& key, Args&&... args) -> std::pair<local_iterator, bool>
	{
		const auto hash = m_hasher(this->to_key(key));
		const auto pos = this->empty() ? m_metatable_end : this->do_find(hash, this->to_key(key));
		if (pos != m_metatable_end) {
			return {local_iterator(this, pos), false};
		}
		return {local_iterator(this, this->do_emplace(hash, std::forward<F>(key), std::forward<Args>(args)...)), true};
	}

	/**
	 * Erase element at @p rm.
	 * @post All references and iterators are invalidated.
	 * @return local_iterator to a unspecified element or end().
	 *         end() can be returned before the container is empty.
	 * @pre @p rm must be a dereferenceable.
	 */
	auto erase(local_iterator rm) -> local_iterator
	{
		this->do_erase(const_cast<meta_t*>(rm.slot()));
		return rm.slot()->dfh_hash ? rm : local_iterator(this, m_metatable_end);
	}

	/**
	 * Erase a element equivalent to @p kval
	 * @post All references and iterators are invalidated if non-zero value is returned.
	 */
	auto erase(const Key& kval) -> size_type
	{
		if (this->empty()) {
			return 0;
		}
		const auto pos = this->do_find(m_hasher(kval), kval);
		if (pos == m_metatable_end) {
			return 0;
		}
		this->do_erase(const_cast<meta_t*>(pos));
		return 1;
	}

	/**
	 * Erase element at @p rm.
	 * @post All references and iterators are invalidated.
	 * @return value_iterator to an adjacent element of @p rm (before or after) or end().
	 *         end() is returned if container was emptied.
	 * @pre @p rm must be a dereferenceable.
	 */
	auto erase(value_iterator rm) -> value_iterator
	{
		size_t dist = std::distance(m_elements.data(), rm.value());
		this->do_erase(const_cast<meta_t*>(rm.slot()));

		if (dist > m_elements.size())
			--dist;
		return value_iterator(this, m_elements.data() + dist);
	}

	/**
	 * Find @kval in the container.
	 */
	template <typename F>
	[[nodiscard]] auto find(F&& kval) -> local_iterator
	{
		auto pos = empty() ? m_metatable_end : this->do_find(m_hasher(this->to_key(kval)), std::forward<F>(kval));
		return local_iterator(this, pos);
	}

	template <typename F>
	[[nodiscard]] auto find(F&& kval) const -> const_local_iterator
	{
		auto pos = empty() ? m_metatable_end : this->do_find(m_hasher(this->to_key(kval)), std::forward<F>(kval));
		return const_local_iterator(this, pos);
	}

	[[nodiscard]] auto contains(const Key& kval) const
	{
		if (this->empty()) {
			return false;
		}
		return this->do_find(m_hasher(this->to_key(kval)), std::forward<const Key>(kval)) != m_metatable_end;
	}

	[[nodiscard]] size_type count(const Key& kval) const
	{
		if (this->empty()) {
			return 0;
		}
		return this->do_find(m_hasher(this->to_key(kval)), std::forward<const Key>(kval)) != m_metatable_end;
	}

	[[nodiscard]] auto empty() const
	{
		return m_elements.empty();
	}

	/**
	 * Iterator to elements.
	 */

	[[nodiscard]] auto begin()
	{
		return value_iterator(this, m_elements.data());
	}

	[[nodiscard]] auto begin() const
	{
		return const_value_iterator(this, m_elements.data());
	}

	[[nodiscard]] auto cbegin() const
	{
		return const_value_iterator(this, m_elements.data());
	}

	/**
	 * Past-end iterator.
	 * @note A past-end local_iterator and value_iterator compare equal.
	 *       A local_iterator can implicitly converted into a value_iterator.
	 */

	[[nodiscard]] auto end()
	{
		return local_iterator(this, m_metatable_end);
	}

	[[nodiscard]] auto end() const
	{
		return const_local_iterator(this, m_metatable_end);
	}

	[[nodiscard]] auto cend() const
	{
		return const_local_iterator(this, m_metatable_end);
	}

	[[nodiscard]] auto vend()
	{
		return value_iterator(this, m_elements.data() + m_elements.size());
	}

	[[nodiscard]] auto vend() const
	{
		return const_value_iterator(this, m_elements.data() + m_elements.size());
	}

	[[nodiscard]] auto vcend() const
	{
		return const_value_iterator(this, m_elements.data() + m_elements.size());
	}

	void clear()
	{
		m_elements.clear();
		std::fill(m_metatable, const_cast<meta_t*>(m_metatable_end), meta_t{0,0});
	}

	/**
	 * Reduce container memory usage.
	 * @post All references and iterators are invalidated.
	 */
	void shrink_to_fit()
	{
		m_elements.shrink_to_fit();
		if (m_elements.empty()) {
			// Reset.
			delete[] m_metatable;
			m_metatable_end = m_metatable = nullptr;
			m_shift = 0;
			m_max_fill = 0;
		} else if (m_elements.size() < size_limit_low()) {
			// Too sparse/wasting memory.
			resize_and_clear(m_elements.size());
			do_rehash();
		}
	}

	void max_load_factor(float lf)
	{
		m_max_load_factor = size_type(lf * 1000);
		m_max_fill = max_fill_from_pow2(mcapacity());
	}

	[[nodiscard]] float max_load_factor() const
	{
		return m_max_load_factor / 1000.0;
	}

	[[nodiscard]] float load_factor() const
	{
		return float(size()) / bucket_count();
	}

	void rehash(size_t new_size)
	{
		new_size = std::max(new_size,  m_elements.size());
		if (calculate_shift_for(new_size) > m_shift) {
			resize_and_clear(new_size);
			do_rehash();
		}
	}

	[[nodiscard]] size_type bucket_count() const noexcept
	{
		return mcapacity();
	}

	[[nodiscard]] size_type size() const noexcept
	{
		return m_elements.size();
	}

	[[nodiscard]] auto hash_function() const
	{
		return m_hasher;
	}

	[[nodiscard]] auto key_eq() const
	{
		return m_equal;
	}

	// TODO: implement more of STL unordered_map API.

	// TODO: sort(): sort the elements in a value iterated range.
	//       This is technically possible thanks the element indexes.
};


template<typename X>
class value_range
{
private:
	X& cont;
public:
	explicit value_range(X & x) : cont(x) {}

	auto begin()
	{
		return cont.begin();
	}

	auto begin() const
	{
		return cont.begin();
	}

	auto cbegin() const
	{
		return cont.begin();
	}

	auto end()
	{
		return cont.vend();
	}

	auto end() const
	{
		return cont.vend();
	}

	auto cend() const
	{
		return cont.vcend();
	}
};

/**
 * Plainly iterating the container with range-for would result in non-optimal code of converting the
 * past-end local_iterator as returned by end() to a value_iterator for the non-equal iterator comparison.
 * The vrange() helper allows using the vend() in a range-for loops.
 */
template<typename X>
auto vrange(X & x) -> value_range<X>
{
	return value_range(x);
}

/**
 * BucketIterator implementation.
 */
template <typename Derived, bool IsConst>
class BucketIterator {
private:
	using derived = Derived;
	using container_type = std::conditional_t<IsConst, const derived, derived>;
	container_type* m_container = nullptr;

protected:
	friend derived;
	using meta_t = typename derived::meta_t;
	using value_t = std::conditional_t<IsConst
									  , typename derived::value_t const
									  , typename derived::value_t>;
	using value_iterator = std::conditional_t<IsConst
											 , typename derived::const_value_iterator
											 , typename derived::value_iterator>;
	friend value_iterator;

private:
	const meta_t* m_itr = nullptr;

protected:

	[[nodiscard]] auto slot() const {
		return m_itr;
	}

	[[nodiscard]] auto value() const
	{
		return m_container->element(m_itr);
	}

	// Only container_type is allowed to construct us.
	explicit BucketIterator(container_type * c, const meta_t* slotptr)
		:  m_container(c), m_itr(slotptr)
	{}

public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = std::conditional_t<IsConst, const value_t, value_t>;
	using reference = std::conditional_t<IsConst, const value_t&, value_t&>;
	using pointer = std::conditional_t<IsConst, const value_t*, value_t*>;

	/// Invalid iterator.
	BucketIterator() =default;

	reference operator*() {
		return *value();
	}

	pointer operator->() {
		return value();
	}

	const pointer operator->() const {
		return value();
	}

	auto operator++() -> auto& {
		m_itr = m_container->advance(m_itr, 1);
		return *this;
	}

	auto operator++(int) -> auto {
		auto tmp = *this;
		m_itr = m_container->advance(m_itr, 1);
		return tmp;
	}

	friend bool operator==(const BucketIterator& a, const BucketIterator& b)
	{
		return a.m_itr == b.m_itr;
	}

	friend bool operator!=(const BucketIterator& a, const BucketIterator& b) {
		return !(a == b);
	}

	bool operator==(const value_iterator& other) const
	{
		// It is cheaper to compare the element's address.
		// m_container->element() has a special case for m_itr == capacity()
		return value() == other.value();
	}

	bool operator!=(const value_iterator& b) const
	{
		return !(*this == b);
	}

	operator value_iterator() const
	{
		return value_iterator(m_container, m_container->element(m_itr));
	}

	[[nodiscard]] value_iterator as_value_iterator() const
	{
		return *this;
	}
};

/**
 * ValueIterator implementation.
 */
template <typename Derived, bool IsConst>
class ValueIterator {
private:
	using derived = Derived;
	using container_type = std::conditional_t<IsConst, const derived, derived>;
	container_type* m_container = nullptr;

protected:
	friend derived;
	using meta_t = typename derived::meta_t;
	using value_t = std::conditional_t<IsConst
									  , typename derived::value_t const
									  , typename derived::value_t>;
	using local_iterator = std::conditional_t<IsConst
											 , typename derived::const_local_iterator
											 , typename derived::local_iterator>;
	friend local_iterator;

private:
	value_t* m_itr;

protected:

	[[nodiscard]] auto slot() const {
		return m_container->convert_iterator(m_itr);
	}

	[[nodiscard]] auto value() const
	{
		return m_itr;
	}

	// container_type and bucket_iterator are allowed to construct us.
	explicit ValueIterator(container_type * c, value_t* itr)
		:  m_container(c), m_itr(itr)
	{}

public:
	using iterator_category = std::random_access_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = std::conditional_t<IsConst, const value_t, value_t>;
	using reference = std::conditional_t<IsConst, const value_t&, value_t&>;
	using pointer = std::conditional_t<IsConst, const value_t*, value_t*>;

	/// Invalid iterator.
	ValueIterator() =default;

	// Forward iterator

	reference operator*() {
		return *value();
	}

	pointer operator->() {
		return value();
	}

	const pointer operator->() const {
		return value();
	}

	auto operator++() -> auto& {
		++m_itr;
		return *this;
	}

	auto operator++(int) -> auto {
		return ValueIterator(m_container, m_itr++);
	}

	friend bool operator==(const ValueIterator& a, const ValueIterator& b)
	{
		return a.m_itr == b.m_itr;
	};

	friend bool operator!=(const ValueIterator& a, const ValueIterator& b) {
		return !(a == b);
	};

	// Bidirectional

	auto operator--() -> auto& {
		--m_itr;
		return *this;
	}

	auto operator--(int) -> auto {
		return ValueIterator(m_container, m_itr--);
	}

	// Random access

	reference operator[] (difference_type n) const {
		return value()[n];
	}

	auto operator+= (difference_type n) -> auto& {
		m_itr += n;
		return *this;
	}

	auto operator+ (difference_type n) const -> auto {
		return ValueIterator(*this) += n;
	}

	auto operator-= (difference_type n) -> auto& {
		m_itr -= n;
		return *this;
	}

	auto operator- (difference_type n) const -> auto {
		return ValueIterator(*this) -= n;
	}

	friend difference_type operator- (const ValueIterator& lhs, const ValueIterator& rhs)
	{
		return std::distance(rhs.value(), lhs.value());
	}

	bool operator==(const local_iterator& other) const
	{
		// It is cheaper to compare the element's address.
		// m_container->element() has a special case for m_itr == end()
		return value() == other.value();
	}

	bool operator!=(const local_iterator& other) const
	{
		return !(*this == other);
	}

	operator local_iterator() const
	{
		return local_iterator(m_container, slot());
	}
};

/**
 * unordered_set<> like container
 */
template <typename Value
		, typename Hasher = std::hash<Value>
		, typename Equal = std::equal_to<Value>>
using RHashSet = HashtableAlgorithms<Value, void, Hasher, Equal>;


/**
 * unordered_map<> like container
 */
template <typename Key
		 , typename T
		 , typename Hasher = std::hash<Key>
		 , typename Equal = std::equal_to<Key>>
using RHashMap = HashtableAlgorithms<Key, T, Hasher, Equal>;

} // <namespace jh>

#endif // CRHASHTABLE_43FF74E8499_H_INCLUDED
