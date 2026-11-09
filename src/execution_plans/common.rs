use datafusion::common::Result;
use datafusion::physical_expr::{Partitioning, RangePartitioning};
use datafusion::physical_plan::PlanProperties;
use std::sync::Arc;

pub(super) fn scale_partitioning_props(
    props: &Arc<PlanProperties>,
    f: impl FnOnce(usize) -> usize,
) -> Result<Arc<PlanProperties>> {
    Ok(Arc::new(PlanProperties::new(
        props.eq_properties.clone(),
        scale_partitioning(&props.partitioning, f)?,
        props.emission_type,
        props.boundedness,
    )))
}

pub(super) fn scale_partitioning(
    partitioning: &Partitioning,
    f: impl FnOnce(usize) -> usize,
) -> Result<Partitioning> {
    match &partitioning {
        Partitioning::RoundRobinBatch(p) => Ok(Partitioning::RoundRobinBatch(f(*p))),
        Partitioning::Hash(hash, p) => Ok(Partitioning::Hash(hash.clone(), f(*p))),
        Partitioning::UnknownPartitioning(p) => Ok(Partitioning::UnknownPartitioning(f(*p))),
        Partitioning::Range(range) => Ok(Partitioning::Range(scale_range_partitioning(
            range,
            f(range.partition_count()),
        )?)),
    }
}

/// Scale a range partitioning to `target` contiguous ranges.
///
/// Range split points are ordered boundaries, so scaling down is deterministic:
/// pick evenly spaced original boundaries.  Scaling up is capped at the current
/// partition count because a range cannot be split without knowing where the
/// underlying keys fall.  The IVM bucket shuffle uses this to merge adjacent
/// LakeSoul buckets when a stage has fewer workers than buckets, while keeping
/// every bucket in exactly one output range.
pub(super) fn scale_range_partitioning(
    range: &RangePartitioning,
    target: usize,
) -> Result<RangePartitioning> {
    let current = range.partition_count();
    let target = target.clamp(1, current);
    if target == current {
        return Ok(range.clone());
    }
    if target == 1 {
        return Ok(RangePartitioning::new(range.ordering().clone(), Vec::new()));
    }
    let split_points = range.split_points();
    let mut scaled = Vec::with_capacity(target - 1);
    for index in 1..target {
        let boundary = index * current / target;
        scaled.push(split_points[boundary - 1].clone());
    }
    RangePartitioning::try_new(range.ordering().clone(), scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::compute::SortOptions;
    use datafusion::common::{ScalarValue, SplitPoint};
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};

    fn range(boundaries: &[u32]) -> RangePartitioning {
        let ordering = LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("bucket", 0)),
            SortOptions::default(),
        )])
        .unwrap();
        let split_points = boundaries
            .iter()
            .map(|value| SplitPoint::new(vec![ScalarValue::UInt32(Some(*value))]))
            .collect();
        RangePartitioning::try_new(ordering, split_points).unwrap()
    }

    #[test]
    fn a_range_scales_down_by_merging_adjacent_partitions() {
        let scaled = scale_range_partitioning(&range(&[1, 2, 3]), 2).unwrap();
        assert_eq!(scaled.partition_count(), 2);
        assert_eq!(scaled.split_points().len(), 1);
        let value = scaled.split_points()[0].values()[0].clone();
        assert_eq!(value, ScalarValue::UInt32(Some(2)));
    }

    #[test]
    fn a_range_never_scales_above_its_bucket_count() {
        let scaled = scale_range_partitioning(&range(&[1, 2, 3]), 8).unwrap();
        assert_eq!(scaled.partition_count(), 4);
    }

    #[test]
    fn a_range_can_collapse_to_one_partition() {
        let scaled = scale_range_partitioning(&range(&[1, 2, 3]), 1).unwrap();
        assert_eq!(scaled.partition_count(), 1);
        assert!(scaled.split_points().is_empty());
    }
}
