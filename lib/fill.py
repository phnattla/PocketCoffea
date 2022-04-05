from coffea import hist
import awkward as ak

def hist_is_in(histname, hist_list, accumulator):
    isHist = type(accumulator[histname]) == hist.Hist
    return ( isHist & (histname in hist_list) )

def fill_histograms_object(processor, obj, obj_hists, event_var=False):
    accumulator = processor.output
    for histname in filter( lambda x : hist_is_in(x, obj_hists, accumulator), accumulator.keys() ):
        h = accumulator[histname]
        for cut in processor._selections.keys():
            if event_var:
                weight = processor.weights.weight() * processor._cuts.all(*processor._selections[cut])
                fields = {k: ak.fill_none(getattr(processor.events, k), -9999) for k in h.fields if k in histname}
            else:
                weight = ak.flatten( processor.weights.weight() * ak.Array(ak.ones_like(obj.pt) * processor._cuts.all(*processor._selections[cut])) )
                fields = {k: ak.flatten(ak.fill_none(obj[k], -9999)) for k in h.fields if k in dir(obj)}
            h.fill(sample=processor._sample, cut=cut, year=processor._year, **fields, weight=weight)