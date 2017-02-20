/*
 * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.cloudera.datascience.intro

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter

case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)
case class Scored(md: MatchData, score: Double)

object RunIntro extends Serializable {
  def main(args: Array[String]): Unit = {
    // 当前可用的最多核数
    val sc = new SparkContext("local[2]", "Intro")
//    val sc = new SparkContext("local[4]", "Intro")
//    val sc = new SparkContext(new SparkConf().setAppName("Intro"))
    val rawblocks = sc.textFile("G:///GithubRepo/aas/data/linkage")
    def isHeader(line: String) = line.contains("id_1")
    
    val noheader = rawblocks.filter(x => !isHeader(x))
    def toDouble(s: String) = {
     if ("?".equals(s)) Double.NaN else s.toDouble
    }

    def parse(line: String) = {
      val pieces = line.split(',')
      val id1 = pieces(0).toInt
      val id2 = pieces(1).toInt
      val scores = pieces.slice(2, 11).map(toDouble)
      val matched = pieces(11).toBoolean
      MatchData(id1, id2, scores, matched)
    }

    val parsed = noheader.map(line => parse(line))
    parsed.cache()

    val matchCounts = parsed.map(md => md.matched).countByValue()
    val matchCountsSeq = matchCounts.toSeq
    matchCountsSeq.sortBy(_._2).reverse.foreach(println)
    /*
    (false,5728201)
    (true,20931)
    */

    // 对每一列(10列)的数据情况进行统计
    val stats = (0 until 9).map(i => {
      parsed.map(_.scores(i)).filter(!_.isNaN).stats()
    })
    stats.foreach(println)
    /*
    (count: 5748125, mean: 0.712902, stdev: 0.388758, max: 1.000000, min: 0.000000)
    (count: 103698, mean: 0.900018, stdev: 0.271316, max: 1.000000, min: 0.000000)
    (count: 5749132, mean: 0.315628, stdev: 0.334234, max: 1.000000, min: 0.000000)
    (count: 2464, mean: 0.318413, stdev: 0.368492, max: 1.000000, min: 0.000000)
    (count: 5749132, mean: 0.955001, stdev: 0.207301, max: 1.000000, min: 0.000000)
    (count: 5748337, mean: 0.224465, stdev: 0.417230, max: 1.000000, min: 0.000000)
    (count: 5748337, mean: 0.488855, stdev: 0.499876, max: 1.000000, min: 0.000000)
    (count: 5748337, mean: 0.222749, stdev: 0.416091, max: 1.000000, min: 0.000000)
    (count: 5736289, mean: 0.005529, stdev: 0.074149, max: 1.000000, min: 0.000000)
    */
    val nasRDD = parsed.map(md => {
      md.scores.map(d => NAStatCounter(d))
    })
    val reduced = nasRDD.reduce((n1, n2) => {
      n1.zip(n2).map { case (a, b) => a.merge(b) }
    })
    reduced.foreach(println)
    /*
    stats: (count: 5748125, mean: 0.712902, stdev: 0.388758, max: 1.000000, min: 0.000000) NaN: 1007
    stats: (count: 103698, mean: 0.900018, stdev: 0.271316, max: 1.000000, min: 0.000000) NaN: 5645434
    stats: (count: 5749132, mean: 0.315628, stdev: 0.334234, max: 1.000000, min: 0.000000) NaN: 0
    stats: (count: 2464, mean: 0.318413, stdev: 0.368492, max: 1.000000, min: 0.000000) NaN: 5746668
    stats: (count: 5749132, mean: 0.955001, stdev: 0.207301, max: 1.000000, min: 0.000000) NaN: 0
    stats: (count: 5748337, mean: 0.224465, stdev: 0.417230, max: 1.000000, min: 0.000000) NaN: 795
    stats: (count: 5748337, mean: 0.488855, stdev: 0.499876, max: 1.000000, min: 0.000000) NaN: 795
    stats: (count: 5748337, mean: 0.222749, stdev: 0.416091, max: 1.000000, min: 0.000000) NaN: 795
    stats: (count: 5736289, mean: 0.005529, stdev: 0.074149, max: 1.000000, min: 0.000000) NaN: 12843
    */

    val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
    val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))
    statsm.zip(statsn).map { case(m, n) =>
      (m.missing + n.missing, m.stats.mean - n.stats.mean)
    }.foreach(println)
    /*
    (1007,0.2854529057466859)
    (5645434,0.09104268062279897)
    (0,0.6838772482597569)
    (5746668,0.8064147192926266)
    (0,0.03240818525033473)
    (795,0.7754423117834044)
    (795,0.5109496938298719)
    (795,0.7762059675300521)
    (12843,0.9563812499852178)
    */
    def naz(d: Double) = if (Double.NaN.equals(d)) 0.0 else d
    val ct = parsed.map(md => {
      val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
      Scored(md, score)
    })

    ct.filter(s => s.score >= 4.0).
      map(s => s.md.matched).countByValue().foreach(println)
    ct.filter(s => s.score >= 2.0).
      map(s => s.md.matched).countByValue().foreach(println)
  }
    /*
    (true,20871)
    (false,637)
    */
  def statsWithMissing(rdd: RDD[Array[Double]]): Array[NAStatCounter] = {
    val nastats = rdd.mapPartitions((iter: Iterator[Array[Double]]) => {
      val nas: Array[NAStatCounter] = iter.next().map(d => NAStatCounter(d))
      iter.foreach(arr => {
        nas.zip(arr).foreach { case (n, d) => n.add(d) }
      })
      Iterator(nas)
    })
    nastats.reduce((n1, n2) => {
      n1.zip(n2).map { case (a, b) => a.merge(b) }
    })
  }
}

class NAStatCounter extends Serializable {
  val stats: StatCounter = new StatCounter()
  var missing: Long = 0

  def add(x: Double): NAStatCounter = {
    if (x.isNaN) {
      missing += 1
    } else {
      stats.merge(x)
    }
    this
  }

  def merge(other: NAStatCounter): NAStatCounter = {
    stats.merge(other.stats)
    missing += other.missing
    this
  }

  override def toString: String = {
    "stats: " + stats.toString + " NaN: " + missing
  }
}

object NAStatCounter extends Serializable {
  def apply(x: Double) = new NAStatCounter().add(x)
}
