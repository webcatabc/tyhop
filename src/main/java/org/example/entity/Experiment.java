package org.example.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

/** MyBatis-Plus entity for the experiment table. */
@TableName("experiment")
public class Experiment {
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    private String name;
    private Long randSeed;
    private Integer num;
    private Integer constructedNum;
    private Integer finishNum;
    private Integer hotpotqaNum;
    private Integer musiqueNum;
    /** Java cannot use a field name beginning with a digit; map it explicitly. */
    @TableField("2wikimultihopqa_num")
    private Integer twoWikiMultihopqaNum;
    private Double f1;
    private Double em;
    private Double hit;
    private Double hotpotqaF1;
    private Double hotpotqaEm;
    private Double hotpotqaHit;
    private Double musiqueF1;
    private Double musiqueEm;
    private Double musiqueHit;
    @TableField("2wikimultihopqa_f1")
    private Double twoWikiMultihopqaF1;
    @TableField("2wikimultihopqa_em")
    private Double twoWikiMultihopqaEm;
    @TableField("2wikimultihopqa_hit")
    private Double twoWikiMultihopqaHit;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public Long getRandSeed() { return randSeed; }
    public void setRandSeed(Long randSeed) { this.randSeed = randSeed; }
    public Integer getNum() { return num; }
    public void setNum(Integer num) { this.num = num; }
    public Integer getConstructedNum() { return constructedNum; }
    public void setConstructedNum(Integer constructedNum) { this.constructedNum = constructedNum; }
    public Integer getFinishNum() { return finishNum; }
    public void setFinishNum(Integer finishNum) { this.finishNum = finishNum; }
    public Integer getHotpotqaNum() { return hotpotqaNum; }
    public void setHotpotqaNum(Integer hotpotqaNum) { this.hotpotqaNum = hotpotqaNum; }
    public Integer getMusiqueNum() { return musiqueNum; }
    public void setMusiqueNum(Integer musiqueNum) { this.musiqueNum = musiqueNum; }
    public Integer getTwoWikiMultihopqaNum() { return twoWikiMultihopqaNum; }
    public void setTwoWikiMultihopqaNum(Integer twoWikiMultihopqaNum) { this.twoWikiMultihopqaNum = twoWikiMultihopqaNum; }
    public Double getF1() { return f1; }
    public void setF1(Double f1) { this.f1 = f1; }
    public Double getEm() { return em; }
    public void setEm(Double em) { this.em = em; }
    public Double getHit() { return hit; }
    public void setHit(Double hit) { this.hit = hit; }
    public Double getHotpotqaF1() { return hotpotqaF1; }
    public void setHotpotqaF1(Double hotpotqaF1) { this.hotpotqaF1 = hotpotqaF1; }
    public Double getHotpotqaEm() { return hotpotqaEm; }
    public void setHotpotqaEm(Double hotpotqaEm) { this.hotpotqaEm = hotpotqaEm; }
    public Double getHotpotqaHit() { return hotpotqaHit; }
    public void setHotpotqaHit(Double hotpotqaHit) { this.hotpotqaHit = hotpotqaHit; }
    public Double getMusiqueF1() { return musiqueF1; }
    public void setMusiqueF1(Double musiqueF1) { this.musiqueF1 = musiqueF1; }
    public Double getMusiqueEm() { return musiqueEm; }
    public void setMusiqueEm(Double musiqueEm) { this.musiqueEm = musiqueEm; }
    public Double getMusiqueHit() { return musiqueHit; }
    public void setMusiqueHit(Double musiqueHit) { this.musiqueHit = musiqueHit; }
    public Double getTwoWikiMultihopqaF1() { return twoWikiMultihopqaF1; }
    public void setTwoWikiMultihopqaF1(Double value) { this.twoWikiMultihopqaF1 = value; }
    public Double getTwoWikiMultihopqaEm() { return twoWikiMultihopqaEm; }
    public void setTwoWikiMultihopqaEm(Double value) { this.twoWikiMultihopqaEm = value; }
    public Double getTwoWikiMultihopqaHit() { return twoWikiMultihopqaHit; }
    public void setTwoWikiMultihopqaHit(Double value) { this.twoWikiMultihopqaHit = value; }
}
