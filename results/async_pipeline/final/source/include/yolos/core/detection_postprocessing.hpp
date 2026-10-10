#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace yolos::post {
// Shared wire format. Coordinates remain in model space until after NMS.
struct Candidate {
    float x1, y1, x2, y2, score;
    int32_t classId;
    uint32_t sourceIndex;  // original prediction index; deterministic tie breaker
    uint32_t reserved{0};
};
static_assert(sizeof(Candidate) == 32, "Candidate wire format must be 32 bytes");
struct PacketHeader {
    uint32_t candidateCount, resultCount, overflow, invalid;
};
static_assert(sizeof(PacketHeader) == 16, "Packet header must be 16 bytes");

inline bool precedes(const Candidate& a, const Candidate& b) {
    return a.score != b.score ? a.score > b.score : a.sourceIndex < b.sourceIndex;
}
inline float iou(const Candidate& a, const Candidate& b) {
    float inter = std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                  std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    float area = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1)-inter;
    return area > 0 ? inter/area : 0.f;
}
inline std::vector<Candidate> collect(const float* raw, int boxes, int classes, float conf) {
    std::vector<Candidate> out; out.reserve(512);
    for (int d=0;d<boxes;++d) {
        int cls=0; float score=raw[4*boxes+d];
        for (int c=1;c<classes;++c) {
            float v=raw[(4+c)*boxes+d];
            if (v>score) {score=v;cls=c;}  // ties select the lowest class ID
        }
        if (!std::isfinite(score)) throw std::runtime_error("Non-finite best class score");
        if (score<=conf) continue;
        float cx=raw[d],cy=raw[boxes+d],w=raw[2*boxes+d],h=raw[3*boxes+d];
        Candidate a{cx-w*.5f,cy-h*.5f,cx+w*.5f,cy+h*.5f,score,cls,uint32_t(d),0};
        if (!std::isfinite(a.x1)||!std::isfinite(a.y1)||!std::isfinite(a.x2)||!std::isfinite(a.y2))
            throw std::runtime_error("Non-finite candidate coordinates");
        out.push_back(a);
    }
    return out;
}
inline std::vector<Candidate> nms(std::vector<Candidate> candidates, float threshold, int maxDet) {
    std::sort(candidates.begin(),candidates.end(),precedes);
    // Preserve the old CPU path's declared max_nms=30000. Accelerated modes
    // reject Engines with >30000 predictions, so they never truncate candidates.
    if (candidates.size()>30000) candidates.resize(30000);
    std::vector<Candidate> kept; kept.reserve(std::min(size_t(maxDet),candidates.size()));
    for (const auto& a:candidates) {
        bool suppressed=false;
        for (const auto& b:kept) if (a.classId==b.classId && iou(a,b)>threshold) {suppressed=true;break;}
        if (!suppressed) kept.push_back(a);
        if (kept.size()>=size_t(maxDet)) break;
    }
    return kept;
}
} // namespace yolos::post
