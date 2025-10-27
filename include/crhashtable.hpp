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
 * - 64-bit metadata per element.
 * - Does probe hash bit stealing. "ultra dense"
 *
 * 64-bit hash function is required. Non-avalanching or lower than 64 bits can severely disable performance.
 *
 * The table uses element indexes into the actual element container, such that robin-hood induced movement
 * of the elements themselves is nearly eliminated. When erasing element, the hash-value is computed for
 * the last element in the array and the slot of this at-end element is searched to do an ERASE(). This
 * element is moved into the erased element, and the metadata is updated. ERASE() operation therefore
 * invalidates all references and iterators to elements.
 *
 * The container uses an local_iterator type for emplace(), find() and erase(). The local_iterator is not
 * optimal for iterating through the container since it iterates over metadata slots. The local_iterator is
 * however efficiently convertible to a value_iterator that provides iteration over the container's element
 * array. The past-end local_iterator and value_iterator compare equal.
 *
 * The begin() and end() return a value_iterator that provides fast random-access iteration over all
 * elements. A value_iterator can be converted back into a local_iterator. This however requires finding
 * the metadata slot of the pointed-to element. This conversion is still more efficient than defererencing
 * the value_iterator and using find().
 *
 * Probe hash bit stealing: INSERT() dynamically balances the count of probe hash bits and DFH bits. More
 * distance-from-home (DFH) bits can be gained in exchange of stealing bits from the probe hashes. The
 * metadata table can thus tolerate extreme collision chain lengths before a rehash is forced. The table
 * therefore uses maximum possible number of bits for the probe hashes, and minimum required number of DFH
 * bits to reduce key comparisons made when probing.
 *
 * The one-way grow_dfh_bits() is much faster than doing full rehash in the event DFH bits are exhausted.
 * However, the non-const probe hash bit-width comes at a slight cost compared to having fixed-sized probe
 * hash field.
 *
 */

namespace jh {

/**
 * Declare primary container template.
 */
template <typename, typename, typename, typename>
class HashTableAlgorithms;

/**
 * Key of value.
 * Element's key is accessed via KeyOf()(value)
 * The functor must yield a type that is accepted by both
 * the equal comparison and the hash functor.
 * @note The container's user must guarantee the key value is not changed
 *       while it is inserted.
 */
template <typename Key>
class IdentityKey {
public:
	using type = Key;
	const type& operator()(const Key& key) const {
		return key;
	}
};

template<typename Key, typename Mapped>
struct KeyOfPair {
	using type = Key;
	const type& operator()(const std::pair<Key,Mapped>& v) const {
		return v.first;
	}
};


/**
 * Bucket iterator template.
 */
template<typename>
class BucketIterator;

/**
 * Value iterator template.
 */
template<typename>
class ValueIterator;

/**
 * Hashtable algorithms.
 * @brief the implementation of RHash[Set,Map]
 */
template <typename Value, typename Hasher, typename Equal, typename KeyOf>
class HashtableAlgorithms {
protected:

	using value_t = Value;
	using keyof_t = KeyOf;
	using key_t = typename keyof_t::type;
	using hasher_t = Hasher;
	using equal_t = Equal;

	friend class BucketIterator<HashtableAlgorithms>;
	friend class ValueIterator<HashtableAlgorithms>;

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

private:
	static constexpr float MAX_LOAD_FACTOR = 0.8f;
	static constexpr slot_type INITIAL_DFH_BITS = 4;
	static constexpr slot_type SLOT_BITS = 8 * sizeof(slot_type);
	// 2^shift capacity.
	uint32_t m_shift = 0;
	// Count of DFH bits.
	uint32_t m_dfh_shift = 0;
	// Max count of elements before an rehash takes place.
	size_t m_max_fill = 0;

	// Metadata table.
	std::unique_ptr<meta_t[]> m_metatable;

	// Actual element data.
	using elements_t = std::vector<value_t>;
	elements_t m_elements;
	hasher_t m_hasher;
	equal_t m_equal;

protected:

	/**
	 * Metadata access.
	 */

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
	inline slot_type dfh(slot_type x) const noexcept
	{
		return x >> (SLOT_BITS - m_dfh_shift);
	}

	// The DFH bits are arranged such that less-than comparison of the whole slot_type
	// value works regardless of what the probe hash bits are.
	inline slot_type dfh_increment() const noexcept
	{
		return slot_type(1) << (SLOT_BITS - m_dfh_shift);
	}

	// Can dfh_hash be incremented by dfh_increment()?
	inline bool dfh_at_max(slot_type dfh_hash) const noexcept
	{
		return dfh_hash > (~dfh_increment());
	}

	inline slot_type probe_hash(size_t hash_value) const noexcept
	{
		// Take SLOT_BITS - INITIAL_DFH_BITS num low bits
		// and integer scale down to probe hash range.
		return (hash_value & 0xFFFFFFFul) * (dfh_increment() - 1) / 0xFFFFFFFul;
	}

	inline size_t mcapacity() const noexcept
	{
		return size_t(1) << m_shift;
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
		if (index + 1 == mcapacity()) [[unlikely]] {
			index = 0;
		} else {
			++index;
		}
		ptr = &m_metatable[index];
	}

	inline size_t size_limit() const noexcept
	{
		return m_max_fill;
	}

	inline size_t size_limit_low() const noexcept
	{
		return m_max_fill / 2;
	}

	void grow_and_clear()
	{
		// Grow capacity and throw away the metadata.
		++m_shift;
		m_max_fill = mcapacity() * MAX_LOAD_FACTOR;

		m_dfh_shift = std::max(m_dfh_shift - 1, INITIAL_DFH_BITS);

		// Delete first and then allocate to minimize peak memory usage.
		m_metatable.reset();
		m_metatable = std::make_unique<meta_t[]>(mcapacity());
		std::fill_n(m_metatable.get(), mcapacity(), meta_t{0,0});
	}

	static constexpr size_t calculate_shift_for(size_t max_elements) noexcept
	{
		size_t shift = 1;
		while (static_cast<size_t>((size_t(1) << shift) * MAX_LOAD_FACTOR) < max_elements) {
			++shift;
		}
		return shift;
	}

	/// Resize for at least @p capacity keys.
	void resize_and_clear(size_t max_elements)
	{
		// Set capacity and throw away the metadata.
		m_shift = calculate_shift_for(max_elements);
		m_max_fill = mcapacity() * MAX_LOAD_FACTOR;

		m_dfh_shift = INITIAL_DFH_BITS;

		m_metatable.reset();
		m_metatable = std::make_unique<meta_t[]>(mcapacity());
		std::fill_n(m_metatable.get(), mcapacity(), meta_t{0,0});
	}

	inline slot_type grow_dfh_bits(slot_type dfh_hash) noexcept
	{
		const uint64_t curr_mask = dfh_increment() - 1u;
		const uint64_t next_mask = curr_mask >> 1;

		for (auto* bucket = &at(0); bucket != &at(mcapacity()); ++bucket) {
			if (bucket->dfh_hash) {
				// integer scale down the probe hashes and right shift the DFH.
				const auto phash = uint64_t(bucket->dfh_hash & curr_mask) * next_mask / curr_mask;
				const auto pdfh = bucket->dfh_hash & (~curr_mask);
				bucket->dfh_hash = (pdfh >> 1) + phash;
			}
		}

		++m_dfh_shift;
		assert(m_dfh_shift < SLOT_BITS);

		return ((dfh_hash & (~curr_mask)) >> 1) + (uint64_t(dfh_hash & curr_mask) * next_mask / curr_mask);
	}

	inline slot_iterator insert_index(size_t hash, index_type elementIndex) noexcept
	{
		slot_iterator hindex = modulus(hash);

		// Metadata entry to be inserted:
		// DFH == 1 => element is at it's home index
		slot_type dfh_inc = dfh_increment();
		meta_t curr = {dfh_inc + probe_hash(hash), elementIndex};

		// Find insertion point.
		auto* bucket = &at(hindex);
		while (curr.dfh_hash <= bucket->dfh_hash) {
			assert(!dfh_at_max(curr.dfh_hash));
			curr.dfh_hash += dfh_inc;
			this->next_bucket(hindex, bucket);
		}

		const slot_iterator pos = hindex;

		// Rich slots are shifted forwards to make space for the poor.
		for(;;) {
			std::swap(*bucket, curr);
			if (!curr.dfh_hash) {
				break;
			} else if (dfh_at_max(curr.dfh_hash)) {
				// ran out of DFH bits.
				curr.dfh_hash = grow_dfh_bits(curr.dfh_hash);
				dfh_inc = dfh_increment();
			}
			curr.dfh_hash += dfh_inc;
			this->next_bucket(hindex, bucket);
		}

		return pos;
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
		const slot_type dfh_dec = dfh_increment();
		auto* bucket = &at(itr);
		auto* prev = bucket;
		this->next_bucket(itr, bucket);

		while (bucket->dfh_hash >= dfh_dec * 2) {
			prev->dfh_hash = bucket->dfh_hash - dfh_dec;
			prev->nth = bucket->nth;
			prev = bucket;
			this->next_bucket(itr, bucket);
		}
		*prev = {0, 0};
	}

	/**
	 * Element access.
	 */

	/// transform const KValue& to const key_t&
	template<typename KValue>
	static constexpr auto to_key(const KValue& value) -> auto
	{
		if constexpr (std::is_same_v<KValue, value_t>) {
			constexpr keyof_t conv;
			return conv(value);
		} else if constexpr (std::is_same_v<KValue, key_t>) {
			return value;
		} else if constexpr (std::is_convertible_v<KValue, key_t>) {
			return value;
		} else {
			static_assert(false, "Your key type is wrong!");
		}
	}

	value_t* element(slot_iterator pos)
	{
		if (pos == mcapacity()) {
			return &*m_elements.end();
		}
		return &m_elements[at(pos).nth];
	}

	value_t* element(slot_iterator pos) const
	{
		if (pos == mcapacity()) {
			return const_cast<value_t*>(&*m_elements.end());
		}
		return const_cast<value_t*>(&m_elements[at(pos).nth]);
	}

	elements_t::iterator convert_iterator(slot_iterator itr) const
	{
		if (itr == mcapacity()) {
			return const_cast<elements_t&>(m_elements).end();
		}
		return const_cast<elements_t&>(m_elements).begin() + at(itr).nth;
	}

	slot_iterator convert_iterator(elements_t::iterator itr) const
	{
		if (itr == m_elements.end()) {
			return mcapacity();
		}
		auto hash = m_hasher(to_key(*itr));
		index_type pos = std::distance(m_elements.data(), const_cast<const value_t*>(&*itr));
		return find_index(hash, pos);
	}

	slot_iterator advance(slot_iterator itr, ptrdiff_t n) const
	{
		// Advance up to end_bucket(), skipping empty buckets.
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
			last = insert_index(m_hasher(to_key(m_elements[i])), i);
		}
		return last;
	}

	/**
	 * INSERT() a value, rehash if necessary.
	 * @return slot_iterator of the inserted m_elements.back()
	 */
	template <typename Key, typename... Args>
	slot_iterator do_emplace(Key&& item, Args&&... args)
	{
		auto& kv = m_elements.emplace_back(std::forward<Key>(item), std::forward<Args>(args)...);

		if (!m_metatable) [[unlikely]] {
			resize_and_clear(8);
		} else if (m_elements.size() >= size_limit()) [[unlikely]] {
			// load-factor is too high => rehash.
			grow_and_clear();
			return do_rehash();
		}

		return insert_index(m_hasher(to_key(kv)), m_elements.size() - 1);
	}

	/**
	 * FIND()
	 */
	template <typename Key>
	slot_iterator do_find(const Key& kval) const
	{
		const size_t hash = m_hasher(to_key(kval));
		const slot_type dfh_inc = dfh_increment();
		slot_type dist = dfh_inc + probe_hash(hash);
		slot_iterator pos = modulus(hash);
		const auto* bucket = &const_at(pos);

		for (;;) {
			if (bucket->dfh_hash == dist) {
				if (m_equal(to_key(m_elements[bucket->nth]), to_key(kval))) {
					// Found!
					return pos;
				}
			} else if (bucket->dfh_hash < dist) {
				// DFH is greater than inserted key can have => key not found.
				break;
			} else if (dfh_at_max(dist)) {
				// reached max DFH.
				break;
			}
			this->next_bucket(pos, bucket);
			dist += dfh_inc;
		}
		return mcapacity();
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
			const auto end_hash = m_hasher(to_key(m_elements.back()));
			const auto rm = find_index(end_hash, at_end);
			at(rm).nth = index;
			// Move the element.
			m_elements[index] = std::move(m_elements.back());
		}

		// Pop-back.
		m_elements.pop_back();

#if 0
		if (at_end < size_limit_low()) [[unlikely]] {
			// Too sparse/wasting memory.
			resize_and_clear(at_end);
			do_rehash();
			return;
		}
#endif
		erase_index(itr);
	}

	void swap(HashtableAlgorithms & cont) noexcept {
		using std::swap;
		std::swap(m_shift, cont.m_shift);
		std::swap(m_dfh_shift, cont.m_dfh_shift);
		std::swap(m_max_fill, cont.m_max_fill);
		std::swap(m_metatable, cont.m_metatable);
		std::swap(m_elements, cont.m_elements);
		std::swap(m_hasher, cont.m_hasher);
		std::swap(m_equal, cont.m_equal);
	}

public:

	using key_type = const key_t;
    using value_type = value_t;
    // todo using mapped_type = value_t::second_type if instantiated HashMap.
    using hasher = hasher_t;
    using key_equal = equal_t;
    // todo using allocator_type = ;

	HashtableAlgorithms() =default;

	HashtableAlgorithms(HashtableAlgorithms && cont) noexcept
		: m_shift(std::exchange(cont.m_shift, 0))
		, m_dfh_shift(std::exchange(cont.m_dfh_shift, 0))
		, m_max_fill(std::exchange(cont.m_max_fill, 0))
		, m_metatable(std::move(cont.m_metatable))
		, m_elements(std::move(cont.m_elements))
		, m_hasher(std::move(cont.m_hasher))
		, m_equal(std::move(cont.m_equal))
	{}

	HashtableAlgorithms(const HashtableAlgorithms & cont)
		: m_shift(cont.m_shift)
		, m_dfh_shift(cont.m_dfh_shift)
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
	 * Construct value from @p key and pass additional arguments to the Value constructor.
	 * @return local_iterator to the inserted element.
	 * @note doesn't check if the key already exist.
	 */
	template <typename F, typename... Args>
	local_iterator emplace(F&& key, Args&&... args)
	{
		return local_iterator(this, this->do_emplace(std::forward<F>(key), std::forward<Args>(args)...));
	}

	/**
	 * Find @p key and if not found construct a new value forwarding arguments to the Value constructor.
	 * @return local_iterator to the found or inserted element.
	 */
	template <typename F, typename... Args>
	std::tuple<local_iterator, bool> emplace_unique(F&& key, Args&&... args)
	{
		if (empty()) {
			return {local_iterator(this, this->do_emplace(std::forward<F>(key), std::forward<Args>(args)...)), true};
		}
		// todo: need do_find_or_emplace()
		auto pos = this->do_find(key);
		if (pos != mcapacity()) {
			return {local_iterator(this, pos), false};
		}
		return {local_iterator(this, this->do_emplace(std::forward<F>(key), std::forward<Args>(args)...)), true};
	}

	/**
	 * Erase element at @p rm.
	 * @post All references and iterators are invalidated.
	 */
	void erase(local_iterator rm)
	{
		this->do_erase(rm.slot());
	}

	/**
	 * Erase element at @p rm.
	 * @post All references and iterators are invalidated.
	 * @return value_iterator to an element or end(). To which element
	 *         the returned non-end() iterator points to is unspecified,
	 * 		   but it is never before rm.
	 */
	value_iterator erase(value_iterator rm)
	{
		size_t dist = std::distance(m_elements.data(), rm.value());
		do_erase(rm.slot());

		if (dist < m_elements.size()) {
			return value_iterator(this, m_elements.begin() + dist);
		} else {
			return end();
		}
	}

	/**
	 * Find @kval in the container.
	 */
	template <typename F>
	local_iterator find(F&& kval)
	{
		if (empty())
			return local_iterator(this, mcapacity());
		return local_iterator(this, this->do_find(std::forward<F>(kval)));
	}

	bool empty() const
	{
		return m_elements.empty();
	}

	/**
	 * Iterator to elements.
	 */
	value_iterator begin() const
	{
		return value_iterator(this, const_cast<elements_t&>(m_elements).begin());
	}

	/**
	 * Past-end iterator.
	 * @note past-end local_iterator and value_iterator compare equal.
	 */
	local_iterator end() const
	{
		return local_iterator(this, mcapacity());
	}

	void clear()
	{
		m_elements.clear();
		std::fill_n(&m_metatable[0], mcapacity(), meta_t{0,0});
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
};


/**
 * BucketIterator implementation.
 */
template <typename Derived>
class BucketIterator {
private:
	using container_type = Derived;
	const container_type* m_container = nullptr;

protected:
	friend container_type;
	using slot_iterator = typename container_type::slot_iterator;
	using element_iter_t = typename container_type::elements_t::iterator;
	using value_iterator = typename container_type::value_iterator;
	friend value_iterator;

	using value_t = typename container_type::value_t;

private:
	slot_iterator m_itr = -1;

protected:

	slot_iterator slot() const {
		return m_itr;
	}

	void advance(ptrdiff_t n = 1)
	{
		m_itr = m_container->advance(m_itr, n);
	}

	value_t* value() const
	{
		return m_container->element(m_itr);
	}

	value_t* value()
	{
		return m_container->element(m_itr);
	}

	// Only container_type is allowed to construct us from a slot_iterator.
	explicit BucketIterator(const container_type * c, size_t index)
		:  m_container(c), m_itr(index)
	{}

public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = value_t;
	using reference = value_t&;
	using pointer = value_t*;

	/// Invalid iterator.
	BucketIterator() =default;

	BucketIterator& operator++() {
		advance();
		return *this;
	}

	BucketIterator operator++(int) {
		BucketIterator tmp = *this;
		advance();
		return tmp;
	}

	/**
	 * All accesses to the value promises to never modify the key of it.
	 */
	reference operator*() {
		return *value();
	}

	pointer operator->() {
		return value();
	}

	const value_type& operator*() const {
		return *value();
	}

	const value_type* operator->() const {
		return value();
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

	operator value_iterator() {
		return value_iterator(m_container, m_container->convert_iterator(m_itr));
	}
};

/**
 * ValueIterator implementation.
 */
template <typename Derived>
class ValueIterator {
private:
	using container_type = Derived;
	const container_type* m_container = nullptr;

protected:
	friend container_type;
	using slot_iterator = typename container_type::slot_iterator;
	using element_iter_t = typename container_type::elements_t::iterator;

	using local_iterator = container_type::local_iterator;
	friend local_iterator;

	using value_t = typename container_type::value_t;
private:
	element_iter_t m_itr;

protected:

	slot_iterator slot() const {
		return m_container->convert_iterator(m_itr);
	}

	void advance(ptrdiff_t n = 1)
	{
		std::advance(m_itr, n);
	}

	value_t* value() const
	{
		return &*m_itr;
	}

	value_t* value()
	{
		return &*m_itr;
	}

	// container_type and bucket_iterator is allowed to construct us.
	explicit ValueIterator(const container_type * c, element_iter_t itr)
		:  m_container(c), m_itr(itr)
	{}

public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = value_t;
	using reference = value_t&;
	using pointer = value_t*;

	/// Invalid iterator.
	ValueIterator() =default;

	ValueIterator& operator++() {
		advance();
		return *this;
	}

	ValueIterator operator++(int) {
		ValueIterator tmp = *this;
		advance();
		return tmp;
	}

	/**
	 * All accesses to the value promises to never modify the key of it.
	 */
	reference operator*() {
		return *value();
	}

	pointer operator->() {
		return value();
	}

	const value_type& operator*() const {
		return *value();
	}

	const value_type* operator->() const {
		return value();
	}

	friend bool operator==(const ValueIterator& a, const ValueIterator& b)
	{
		return a.m_itr == b.m_itr;
	};

	friend bool operator!=(const ValueIterator& a, const ValueIterator& b) {
		return !(a == b);
	};

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

	operator local_iterator() const {
		return bucket_iterator(m_container, m_container->convert_iterator(m_itr));
	}
};

/**
 * unordered_set<> like container
 */
template <typename Value
		, typename Hasher = std::hash<Value>
		, typename Equal = std::equal_to<Value>
		, typename KeyOf = IdentityKey<Value>>
using RHashSet = HashtableAlgorithms<Value, Hasher, Equal, KeyOf>;


/**
 * unordered_map<> like container
 */
template <typename Key
		 , typename T
		 , typename Hasher = std::hash<Key>
		 , typename Equal = std::equal_to<Key>>
using RHashMap = HashtableAlgorithms<std::pair<Key, T>, Hasher, Equal, KeyOfPair<Key, T>>;



} // namespace jh

#endif // CRHASHTABLE_43FF74E8499_H_INCLUDED
