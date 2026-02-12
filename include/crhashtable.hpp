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
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <cmath>

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

	static constexpr const Key& to_key(const element_type& value)
	{
		return const_cast<const Key&>(value.first);
	}

	static constexpr const Key& to_key(const Key& value)
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

	static constexpr const Key& to_key(const Key& value)
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
	// In this version jh::HashtableAlgorithms<> m_dfh_shift is a constant.
	static constexpr uint32_t DFH_SHIFT = SLOT_BITS / 2;
	float m_max_load_factor = 0.8f;
	// 2^shift capacity.
	uint32_t m_shift = 0;
	// Max count of elements before an rehash takes place.
	size_t m_max_fill = 0;

	// Metadata table.
	std::unique_ptr<meta_t[]> m_metatable;

	// Actual element data.
	using value_t = typename Config<Key, Value>::element_type;
	using elements_t = std::vector<typename Config<Key, Value>::store_type>;
	elements_t m_elements;
	hasher_t m_hasher;
	equal_t m_equal;

protected:

	/**
	 * Metadata access.
	 */

	inline size_t mcapacity() const noexcept
	{
		return size_t(1) << m_shift;
	}

	inline meta_t& at(slot_iterator i) noexcept
	{
		return m_metatable[i];
	}

	inline const meta_t& at(slot_iterator i) const noexcept
	{
		return m_metatable[i];
	}

	inline const meta_t& const_at(slot_iterator i) const noexcept
	{
		return m_metatable[i];
	}

	// Distance-from-home
	// Zero is reserved for empty slot so 1 is at home slot.
	static constexpr slot_type dfh(slot_type x) noexcept
	{
		return x >> (SLOT_BITS - DFH_SHIFT);
	}

	// The DFH bits are arranged such that less-than comparison of the whole slot_type
	// value works regardless of what the probe hash bits are.
	static constexpr slot_type dfh_increment() noexcept
	{
		return slot_type(1) << (SLOT_BITS - DFH_SHIFT);
	}

	// Can dfh_hash be incremented by dfh_increment()?
	static constexpr bool dfh_overflow(slot_type dfh_hash) noexcept
	{
		return dfh_hash >= ((~slot_type(0)) << (SLOT_BITS - DFH_SHIFT));
	}

	static constexpr slot_type probe_hash(size_t hash_value) noexcept
	{
		// Grab the low m_dfh_shift bits.
		return hash_value & ((slot_type(1) << DFH_SHIFT) - 1);
	}

	inline size_t modulus(size_t hash_value) const noexcept
	{
		// The low hash bits become part of the slot's probe hash.
		// These bits shouldn't be part of the modulus, thus the modulus
		// is computed via right-shifting the hash bits into the correct range.
		return hash_value >> (sizeof(hash_value) * 8 - m_shift);
	}

	// Advance to next bucket or wraparound.
	template<typename Meta>
	inline void next_bucket(slot_iterator& index, Meta*& ptr) const noexcept
	{
		if (index == mcapacity() - 1) [[unlikely]] {
			index = 0;
		} else {
			++index;
		}
		ptr = &m_metatable[index];
	}

	template<typename Meta>
	inline void next_bucket(Meta*& ptr) const noexcept
	{
		// Index-less.
		if (ptr == m_metatable.get() + mcapacity() - 1) [[unlikely]] {
			ptr = m_metatable.get();
		} else {
			++ptr;
		}
	}

	inline size_t size_limit() const noexcept
	{
		return m_max_fill;
	}

	inline size_t size_limit_low() const noexcept
	{
		return m_max_fill / 2;
	}

	size_t calculate_shift_for(size_t max_elements) const noexcept
	{
		size_t shift = 1;
		while (static_cast<size_t>((size_t(1) << shift) * m_max_load_factor) < max_elements) {
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
		m_max_fill = mcapacity() * m_max_load_factor;

		// Delete first and then allocate to minimize peak memory usage.
		m_metatable.reset();
		// Zero initialized for no extra cost.
		m_metatable = std::make_unique<meta_t[]>(mcapacity());
	}

	/// Resize for at least @p capacity keys.
	void resize_and_clear(size_t max_elements)
	{
		// Set capacity and throw away the metadata.
		m_shift = calculate_shift_for(max_elements);
		m_max_fill = mcapacity() * m_max_load_factor;

		m_metatable.reset();
		// Zero initialized for no extra cost.
		m_metatable = std::make_unique<meta_t[]>(mcapacity());
	}

	inline slot_iterator insert_index(size_t hash, index_type elementIndex) noexcept
	{
		// Metadata entry to be inserted:
		// DFH == 1 => element is at it's home index
		meta_t curr = {dfh_increment() + probe_hash(hash), elementIndex};

		// Find insertion point, i.e. skip over too poor slots.
		auto* bucket = &at(modulus(hash));
		while (curr.dfh_hash <= bucket->dfh_hash) {
			assert(!dfh_overflow(curr.dfh_hash));
			curr.dfh_hash += dfh_increment();
			this->next_bucket(bucket);
		}

		const slot_iterator slot = std::distance(&at(0), bucket);

		// Rich slots are shifted forwards to make space for the poor.
		for(;;) {
			std::swap(*bucket, curr);
			if (!curr.dfh_hash) {
				return slot;
			} else if (dfh_overflow(curr.dfh_hash)) [[unlikely]] {
				// ran out of DFH bits, the metadata now trash.
				return mcapacity();
			}
			curr.dfh_hash += dfh_increment();
			this->next_bucket(bucket);
		}
	}

	inline slot_iterator find_index(size_t hash, uint32_t elementIndex) const noexcept
	{
		slot_iterator pos = modulus(hash);
		const auto* bucket = &const_at(pos);
		// The element must be at bucket or some slot after this.
		// A empty-slot cannot exist before the element's slot.
		while (bucket->nth != elementIndex) {
			assert(bucket->dfh_hash);
			this->next_bucket(pos, bucket);
		}
		assert(bucket->dfh_hash);
		return pos;
	}

	inline void erase_index(slot_iterator itr) noexcept
	{
		// Maintain the robin-hood hashing invariant:
		// Shift backwards from bucket+1 until empty slot or home index is found.
		// The DFH is consequently decremented by one.
		auto* bucket = &at(itr);
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

	value_t* element(slot_iterator pos)
	{
		if (pos == mcapacity()) {
			return m_elements.data() + m_elements.size();
		}
		// For set like containers non-const return type is impossible.
		return &m_elements[at(pos).nth];
	}

	const value_t* element(slot_iterator pos) const
	{
		if (pos == mcapacity()) {
			return m_elements.data() + m_elements.size();
		}
		return &m_elements[at(pos).nth];
	}

	slot_iterator convert_iterator(value_t* itr)
	{
		if (itr == m_elements.data() + m_elements.size()) {
			return mcapacity();
		}
		auto hash = m_hasher(this->to_key(*itr));
		index_type pos = std::distance(m_elements.data(), itr);
		return find_index(hash, pos);
	}

	slot_iterator convert_iterator(const value_t* itr) const
	{
		if (itr == m_elements.data() + m_elements.size()) {
			return mcapacity();
		}
		auto hash = m_hasher(this->to_key(*itr));
		index_type pos = std::distance(m_elements.data(), itr);
		return find_index(hash, pos);
	}

	slot_iterator advance(slot_iterator itr, ptrdiff_t n) const
	{
		// Advance up to end(), skipping empty buckets.
		while (n && itr != mcapacity()) {
			n -= at(itr).dfh_hash != slot_type(0);
			++itr;
		}
		return itr;
	}

	/**
	 * Rehash and re-insert all elements into the Metatable.
	 * @pre metadata table must have been cleared.
	 * @return slot_iterator of the elements.back()
	 */
	slot_iterator do_rehash()
	{
		auto last = mcapacity();

		// Re-build metadata.
		for (size_t i = 0; i < m_elements.size(); ++i) {
			last = insert_index(m_hasher(this->to_key(m_elements[i])), i);
			if (last == mcapacity()) {
				// Oops, too many collisions.
				grow_and_clear();
				i = 0;
			}
		}
		return last;
	}

	/**
	 * INSERT() a value, rehash if necessary.
	 * @return slot_iterator of the inserted m_elements.back()
	 */
	template <typename Item, typename... Args>
	slot_iterator do_emplace(const size_t hash, Item&& item, Args&&... args)
	{
		m_elements.emplace_back(std::forward<Item>(item), std::forward<Args>(args)...);

		if (m_elements.size() >= size_limit()) [[unlikely]] {
			// load-factor is too high => rehash or no metatable is allocated yet.
			grow_and_clear();
			return do_rehash();
		}

		auto itr = insert_index(hash, m_elements.size() - 1);
		if (itr == mcapacity()) {
			grow_and_clear();
			return do_rehash();
		}
		return itr;
	}

	/**
	 * FIND()
	 */
	slot_iterator do_find(const size_t hash, const Key& kval) const
	{
		slot_type dist = dfh_increment() + probe_hash(hash);
		const auto* bucket = &const_at(modulus(hash));

		for (;;) {
			if (bucket->dfh_hash < dist) {
				// DFH is greater than inserted key can have => key not found.
				// This branch is before the below branch, to early return on empty home slots.
				return mcapacity();
			} else if (bucket->dfh_hash == dist) {
				if (m_equal(kval, this->to_key(m_elements[bucket->nth]))) {
					// Found!
					return std::distance(&const_at(0), bucket);
				}
			}
			this->next_bucket(bucket);
			dist += dfh_increment();
		}
	}

	/**
	 * ERASE()
	 */
	void do_erase(slot_iterator itr)
	{
		const auto index = at(itr).nth;
		const slot_iterator at_end = m_elements.size() - 1;

		if (index != at_end) [[likely]] {
			// Erase the element: find where the m_elements.back() is in the metadata table,
			// and redirect it to itr's slot.
			const auto end_hash = m_hasher(this->to_key(m_elements.back()));
			const auto rm = find_index(end_hash, at_end);
			at(rm).nth = index;
			// Move the element.
			this->move_assign(m_elements[index], m_elements.back());
		}

		// Pop-back.
		m_elements.pop_back();
		erase_index(itr);
	}

	void swap(HashtableAlgorithms & cont) noexcept
	{
		using std::swap;
		std::swap(m_max_load_factor, cont.m_max_load_factor);
		std::swap(m_shift, cont.m_shift);
		std::swap(m_max_fill, cont.m_max_fill);
		std::swap(m_metatable, cont.m_metatable);
		std::swap(m_elements, cont.m_elements);
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
		: m_max_load_factor(std::exchange(cont.m_m_max_load_factor, 0))
		, m_shift(std::exchange(cont.m_shift, 0))
		, m_max_fill(std::exchange(cont.m_max_fill, 0))
		, m_metatable(std::move(cont.m_metatable))
		, m_elements(std::move(cont.m_elements))
		, m_hasher(std::move(cont.m_hasher))
		, m_equal(std::move(cont.m_equal))
	{}

	HashtableAlgorithms(const HashtableAlgorithms & cont)
		: m_max_load_factor(cont.m_max_load_factor)
		, m_shift(cont.m_shift)
		, m_max_fill(cont.m_max_fill)
		, m_metatable(new meta_t[cont.mcapacity()])
		, m_elements(cont.m_elements)
		, m_hasher(cont.m_hasher)
		, m_equal(cont.m_equal)
	{
		std::copy_n(cont.m_metatable.get(), cont.mcapacity(), m_metatable.get());
	}

	HashtableAlgorithms& operator= (const HashtableAlgorithms& cont)
	{
		HashtableAlgorithms(cont).swap(*this);
		return *this;
	}

	HashtableAlgorithms& operator= (HashtableAlgorithms&& cont) noexcept
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

	~HashtableAlgorithms() =default;

	/**
	 * Construct value_type from @p key and pass additional arguments to the value_type constructor.
	 * @return local_iterator to the inserted element or to a existing element if the key already exist.
	 */
	template <typename F, typename... Args>
	std::pair<local_iterator, bool> emplace(F&& key, Args&&... args)
	{
		const auto hash = m_hasher(this->to_key(key));
		const auto pos = this->empty() ? mcapacity() : this->do_find(hash, this->to_key(key));
		if (pos != mcapacity()) {
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
	local_iterator erase(local_iterator rm)
	{
		this->do_erase(rm.slot());
		return const_at(rm.slot()).dfh_hash ? rm : local_iterator(this, mcapacity());
	}

	/**
	 * Erase a element equivalent to @p kval
	 * @post All references and iterators are invalidated if non-zero value is returned.
	 */
	size_type erase(const Key& kval)
	{
		const auto pos = this->do_find(m_hasher(kval), kval);
		if (pos == mcapacity()) {
			return 0;
		}
		this->do_erase(pos);
		return 1;
	}

	/**
	 * Erase element at @p rm.
	 * @post All references and iterators are invalidated.
	 * @return value_iterator to an adjacent element of @p rm (before or after) or end().
	 *         end() is returned if container was emptied.
	 * @pre @p rm must be a dereferenceable.
	 */
	value_iterator erase(value_iterator rm)
	{
		size_t dist = std::distance(m_elements.data(), rm.value());
		do_erase(rm.slot());

		if (dist > m_elements.size())
			--dist;
		return value_iterator(this, m_elements.data() + dist);
	}

	/**
	 * Find @kval in the container.
	 */
	template <typename F>
	local_iterator find(F&& kval)
	{
		auto pos = empty() ? mcapacity() : this->do_find(m_hasher(this->to_key(kval)), std::forward<F>(kval));
		return local_iterator(this, pos);
	}

	template <typename F>
	const_local_iterator find(F&& kval) const
	{
		auto pos = empty() ? mcapacity() : this->do_find(m_hasher(this->to_key(kval)), std::forward<F>(kval));
		return local_iterator(this, pos);
	}


	bool contains(const Key& kval) const
	{
		return find(kval) != end();
	}

	size_t count(const Key& kval) const
	{
		return find(kval) == end() ? 0 : 1;
	}

	bool empty() const
	{
		return m_elements.empty();
	}

	/**
	 * Iterator to elements.
	 */

	value_iterator begin()
	{
		return value_iterator(this, m_elements.data());
	}

	const_value_iterator begin() const
	{
		return const_value_iterator(this, m_elements.data());
	}

	const_value_iterator cbegin() const
	{
		return const_value_iterator(this, m_elements.data());
	}

	/**
	 * Past-end iterator.
	 * @note A past-end local_iterator and value_iterator compare equal.
	 *       A local_iterator can implicitly converted into a value_iterator.
	 */

	local_iterator end()
	{
		return local_iterator(this, mcapacity());
	}

	const_local_iterator end() const
	{
		return const_local_iterator(this, mcapacity());
	}

	const_local_iterator cend() const
	{
		return const_local_iterator(this, mcapacity());
	}

	value_iterator vend()
	{
		return value_iterator(this, m_elements.data() + m_elements.size());
	}

	const_value_iterator vend() const
	{
		return const_value_iterator(this, m_elements.data() + m_elements.size());
	}

	const_value_iterator vcend() const
	{
		return const_value_iterator(this, m_elements.data() + m_elements.size());
	}

	void clear()
	{
		m_elements.clear();
		std::fill_n(&m_metatable[0], mcapacity(), meta_t{0,0});
	}

	/**
	 * Reduce container memory usage.
	 * @post All references and iterators are invalidated.
	 */
	void shrink_to_fit()
	{
		m_elements.shrink_to_fit();
		if (m_elements.size() < size_limit_low()) {
			// Too sparse/wasting memory.
			resize_and_clear(m_elements.size());
			do_rehash();
		}
	}

	void max_load_factor(float lf)
	{
		m_max_load_factor = lf;
		m_max_fill = mcapacity() * m_max_load_factor;
	}

	float max_load_factor() const
	{
		return m_max_load_factor;
	}

	float load_factor() const
	{
		return size() / bucket_count();
	}

	void rehash(size_t new_size)
	{
		new_size = std::max(new_size,  m_elements.size());
		if (calculate_shift_for(new_size) > m_shift) {
			resize_and_clear(new_size);
			do_rehash();
		}
	}

	size_t bucket_count() const noexcept
	{
		return mcapacity();
	}

	size_t size() const noexcept
	{
		return m_elements.size();
	}

	Hasher hash_function() const
	{
		return m_hasher;
	}

	Equal key_eq() const
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
	return value_range_t(x);
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
	using slot_iterator = typename derived::slot_iterator;
	using value_t = std::conditional_t<IsConst
									  , typename derived::value_t const
									  , typename derived::value_t>;
	using value_iterator = std::conditional_t<IsConst
											 , typename derived::const_value_iterator
											 , typename derived::value_iterator>;
	friend value_iterator;

private:
	slot_iterator m_itr = -1;

protected:

	slot_iterator slot() const {
		return m_itr;
	}

	value_t* value() const
	{
		return m_container->element(m_itr);
	}

	// Only container_type is allowed to construct us from a slot_iterator.
	explicit BucketIterator(container_type * c, size_t index)
		:  m_container(c), m_itr(index)
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

	BucketIterator& operator++() {
		m_itr = m_container->advance(m_itr, 1);
		return *this;
	}

	BucketIterator operator++(int) {
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

	value_iterator as_value_iterator() const
	{
		return *this;
	}

	// TODO: index() to return the pointed-to element's index in the element array.
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
	using slot_iterator = typename derived::slot_iterator;
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

	slot_iterator slot() const {
		return m_container->convert_iterator(m_itr);
	}

	value_t* value() const
	{
		return m_itr;
	}

	// container_type and bucket_iterator is allowed to construct us.
	explicit ValueIterator(container_type * c, value_t* itr)
		:  m_container(c), m_itr(itr)
	{}

public:
	// TODO: implement actual random access
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

	ValueIterator& operator++() {
		++m_itr;
		return *this;
	}

	ValueIterator operator++(int) {
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

	ValueIterator& operator--() {
		--m_itr;
		return *this;
	}

	ValueIterator operator--(int) {
		return ValueIterator(m_container, m_itr--);
	}

	// Random access

	reference operator[] (difference_type n) const {
		return value()[n];
	}

	ValueIterator& operator+= (difference_type n) {
		m_itr += n;
		return *this;
	}

	ValueIterator operator+ (difference_type n) const {
		return ValueIterator(*this) += n;
	}

	ValueIterator& operator-= (difference_type n) {
		m_itr -= n;
		return *this;
	}

	ValueIterator operator- (difference_type n) const {
		return ValueIterator(*this) -= n;
	}

	bool operator==(const local_iterator& other) const
	{
		// It is cheaper to compare the element's address of other.
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

} // namespace jh

#endif // CRHASHTABLE_43FF74E8499_H_INCLUDED
